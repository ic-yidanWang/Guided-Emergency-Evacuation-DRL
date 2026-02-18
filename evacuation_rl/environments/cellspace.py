"""
Continuum Cell Space Environment for Agent (Particle) Dynamics

This module implements the physical simulation environment for emergency evacuation,
including particle dynamics, cell-based spatial partitioning, and force calculations.
"""

import numpy as np
import os


# Physical parameters (all distances in absolute coordinate units)
f_wall_lim = 100.0                      
f_collision_lim = 100.0                 
door_size = 1.0                         # Door size in absolute units (used for door area calculations)
agent_size = 0.5                        # Agent radius in absolute units
reward = -0.1
end_reward = 0.

offset = np.array([0.5, 0.5])           
dis_lim = 0.05                          # Distance threshold for evacuation in absolute units (particle within this distance to exit is considered evacuated)    
action_force = 1.0                      
desire_velocity = 2.0                   
relaxation_time = 0.5                   
delta_t = 0.1                           
cfg_save_step = 5                       

# Initialize Exit positions (range [0, 1])
Exit = []  # No exit by default

# Initialize obstacles
Ob = []   # No obstacles by default
Ob_size = []

# Initialize Guide positions (range [0, 1])
Guide = []  # No guide by default

# Cell and neighbor list for spatial partitioning
cell_list = np.array([
    [1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0],
    [0, 0, -1], [1, 0, -1], [1, 1, -1], [0, 1, -1],
    [-1, 1, -1], [-1, 0, -1], [-1, -1, -1], [0, -1, -1], [1, -1, -1]
], dtype='int')

neighbor_list = np.array([
    [-1, 1, 0], [0, 1, 0], [1, 1, 0],
    [-1, 0, 0], [0, 0, 0], [1, 0, 0], 
    [-1, -1, 0], [0, -1, 0], [1, -1, 0]
], dtype='int')


class Particle:
    """Represents an individual agent/person in the evacuation simulation"""
    
    def __init__(self, ID, x, y, z, vx, vy, vz, mass=80.0, type=1):
        self.position = np.array((x, y, z))
        self.velocity = np.array((vx, vy, vz))
        self.acc = np.array((0., 0., 0.))
        self.mass = mass
        self.type = type
        self.ID = ID
    
    def leapfrog(self, dt, stage):
        """Leapfrog integration for position and velocity updates"""
        if stage == 0:
            self.velocity += dt / 2 * self.acc
            self.position += dt * self.velocity          
        else:
            self.velocity = self.velocity + dt / 2 * self.acc

    def scale_velocity(self, value=1.0):
        """Scale velocity to a specific magnitude"""
        self.velocity /= np.sqrt(np.sum(self.velocity ** 2))
        self.velocity *= value        
  
      
class Cell:
    """Represents a spatial cell for efficient neighbor finding"""
    
    def __init__(self, ID, idx, idy, idz, d_cells, L, n_cells):
        self.Particles = []  # Particle list to store agents in this cell
        self.Neighbors = []  # Identify and store neighbor cells
        self.ID_number = ID  # ID number of the cell
        self.ID_index = np.array([idx, idy, idz])  # ID index of the cell
        self.L = np.zeros_like(L)  # Lower and upper boundary of the cell
        self.L[:, 0] = L[:, 0] + self.ID_index * d_cells
        self.L[:, 1] = self.L[:, 0] + d_cells
        self.n_cells = n_cells
        
        # Visibility system: tracks exit visibility based on path congestion
        self.exit_visibility = {}  # Maps exit_id to visibility value (0-1)
        self.particle_count_to_exit = {}  # Maps exit_id to particle count on path
        self.path_to_exit = {}  # Maps exit_id to path (list of cell IDs)
        
        self.find_neighbors()   
        
    def add(self, particle):
        """Add a particle to this cell"""
        self.Particles.append(particle)
    
    def find_neighbors(self):
        """Find and set the neighbor cells"""
        idx = self.ID_index + cell_list        
        valid = (idx < self.n_cells) & (idx >= 0)
        idx = idx[np.all(valid, axis=1)]
        
        for n in range(len(idx)):
            i = idx[n, 0]
            j = idx[n, 1]
            k = idx[n, 2]
            
            N = k * (self.n_cells[0] * self.n_cells[1]) + j * self.n_cells[0] + i
            self.Neighbors.append(N)        
    

class Cell_Space:
    """
    Continuum Cell Space - Main simulation environment
    
    This environment simulates emergency evacuation using particle dynamics
    and cell-based spatial partitioning for efficient computation.
    """
    
    def __init__(self, xmin=0., xmax=1., ymin=0., ymax=1.,
                 zmin=0., zmax=1., rcut=0.5, dt=0.01, Number=1):
        
        self.dt = dt
        self.Number = Number    # Current number of agents
        self.Total = Number     # Total number of agents
        self.T = 0.             # Temperature of the system
        
        # Size of the system
        self.L = np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]], dtype=np.float32)
        self.rcut = rcut
        
        # Number of cells in each dimension
        self.n_cells = np.array(((xmax - xmin), (ymax - ymin), (zmax - zmin))) / rcut 
        self.n_cells = self.n_cells.astype('int')
        
        # Cell size in each dimension
        self.d_cells = (self.L[:, 1] - self.L[:, 0]) / self.n_cells

        # Set exit information (Exit coordinates are now in absolute format)
        self.Exit = []
        for e in Exit:            
            self.Exit.append(e)
            
        # Set Obstacles information (Obstacle coordinates are now in absolute format)
        # New optimized structure: stores type and geometry directly
        self.Ob = []
        self.Ob_size = []
        self.Ob_type = []  # 'circle' or 'rectangle'
        self.Ob_params = []  # Additional parameters (e.g., width, height for rectangles)
        
        for idx, ob in enumerate(Ob):  
            tmp = []
            for i in ob:
                tmp.append(i) 
                
            self.Ob.append(tmp)    
            self.Ob_size.append(Ob_size[idx])
            self.Ob_type.append('points')  # Default to points for backward compatibility
            self.Ob_params.append({})
        
        # Set Guide information (Guide coordinates are now in absolute format)
        self.Guide = []
        for g in Guide:
            self.Guide.append(g)

        self.Cells = []
        self.initialize_cells()
        self.initialize_particles()
   
        diag = np.sqrt(2) / 2
        self.action = np.array([
            [0, 1, 0], [-diag, diag, 0], [-1, 0, 0], [-diag, -diag, 0],
            [0, -1, 0], [diag, -diag, 0], [1, 0, 0], [diag, diag, 0]
        ], dtype=np.float32)
        self.action *= action_force
        
        self.reward = reward     
        self.end_reward = end_reward
        
        # Visibility system parameters
        self.visibility_alpha = 0.5  # Controls how strongly particle count affects visibility
        self.max_visibility_distance = 10  # Maximum distance to calculate visibility for
        
        # Initialize visibility system
        self.update_visibility_system()
        
    def initialize_cells(self):
        """Initialize the cell grid"""
        nx = self.n_cells[0]
        ny = self.n_cells[1]
        nz = self.n_cells[2]
        
        np = nx * ny
        n_total = np * nz
        
        for n in range(n_total):
            # Convert to the cell ID index
            i = n % np % nx
            j = n % np // nx
            k = n // np
          
            self.Cells.append(Cell(n, i, j, k, self.d_cells, self.L, self.n_cells))
    
    def get_cell_id_from_position(self, position):
        """Get cell ID from position coordinates"""
        index = (position - self.L[:, 0]) / self.d_cells
        index = np.clip(index, 0, self.n_cells - 1).astype(int)
        cell_id = int(index[2] * (self.n_cells[0] * self.n_cells[1]) + 
                     index[1] * self.n_cells[0] + index[0])
        return max(0, min(cell_id, len(self.Cells) - 1))
    
    def get_nearest_exit_cell_id(self, from_cell_id):
        """Find the nearest exit for a given cell"""
        from_cell_pos = self.Cells[from_cell_id].L[:, 0]  # Cell's lower corner
        
        min_dist = float('inf')
        nearest_exit_id = 0
        
        for exit_id, exit_pos in enumerate(self.Exit):
            dist = np.sqrt(np.sum((from_cell_pos - exit_pos[:self.Cells[0].L.shape[0]]) ** 2))
            if dist < min_dist:
                min_dist = dist
                nearest_exit_id = exit_id
        
        return nearest_exit_id
    
    def bfs_path_to_exit(self, from_cell_id, exit_id):
        """
        Use BFS to find shortest path from a cell to the exit using cell neighbors.
        Returns list of cell IDs forming the path.
        """
        from collections import deque
        
        if exit_id >= len(self.Exit):
            return []
        
        exit_pos = self.Exit[exit_id]
        
        # Find which cell contains or is nearest to the exit
        target_cell_id = self.get_cell_id_from_position(exit_pos)
        
        if from_cell_id == target_cell_id:
            return [from_cell_id]
        
        visited = set()
        queue = deque([(from_cell_id, [from_cell_id])])
        visited.add(from_cell_id)
        
        while queue:
            current_cell_id, path = queue.popleft()
            
            if current_cell_id == target_cell_id:
                return path
            
            current_cell = self.Cells[current_cell_id]
            
            # Explore neighbors (use the cell's neighbor list)
            for neighbor_id in current_cell.Neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
        
        # If no path found, return empty path
        return []
    
    def calculate_path_visibility(self, path, exit_id):
        """
        Calculate the visibility of an exit based on the number of particles on the path.
        visibility = 1.0 / (1.0 + alpha * particle_count)
        Returns: (particle_count, visibility)
        """
        particle_count = 0
        
        for cell_id in path:
            if cell_id < len(self.Cells):
                particle_count += len(self.Cells[cell_id].Particles)
        
        # Calculate visibility: high when few particles, low when many particles
        visibility = 1.0 / (1.0 + self.visibility_alpha * particle_count)
        
        return particle_count, visibility
    
    def update_visibility_system(self):
        """
        Update visibility for all cells based on congestion on paths to exits.
        This should be called each simulation step.
        """
        if not self.Exit:
            return
        
        for cell_id, cell in enumerate(self.Cells):
            if not cell.Particles:
                continue
            
            # For each exit, calculate visibility
            for exit_id in range(len(self.Exit)):
                path = self.bfs_path_to_exit(cell_id, exit_id)
                
                if path:
                    particle_count, visibility = self.calculate_path_visibility(path, exit_id)
                    cell.exit_visibility[exit_id] = visibility
                    cell.particle_count_to_exit[exit_id] = particle_count
                    cell.path_to_exit[exit_id] = path
                else:
                    cell.exit_visibility[exit_id] = 0.0
                    cell.particle_count_to_exit[exit_id] = float('inf')
                    cell.path_to_exit[exit_id] = []
    
    def get_exit_visibility_for_particle(self, particle, exit_id=None):
        """
        Get visibility of exit(s) for a particle.
        If exit_id is None, returns visibility for nearest exit.
        Returns: visibility value (0-1)
        """
        if not self.Exit:
            return 0.0
        
        from_cell_id = self.get_cell_id_from_position(particle.position)
        cell = self.Cells[from_cell_id]
        
        if exit_id is None:
            exit_id = self.get_nearest_exit_cell_id(from_cell_id)
        
        return cell.exit_visibility.get(exit_id, 0.0)
    
    def Zero_acc(self):
        """Reset acceleration for all particles"""
        for c in self.Cells:
            for p in c.Particles:
                p.acc[:] = 0.
                
    def Normalization(self, position):
        """Normalization to [0,1]"""
        return (position - self.L[:, 0]) / (self.L[:, 1] - self.L[:, 0])
    
    def Normalization_XY(self, position, offset=offset):
        """Normalization to [0,1] at xy plane and take offset"""
        return (position - self.L[:2, 0]) / (self.L[:2, 1] - self.L[:2, 0]) - offset
    
    def Integration(self, stage):
        """Integration step using leapfrog method"""
        self.T = 0.
        
        for c in self.Cells:
            for p in c.Particles:
                p.leapfrog(dt=self.dt, stage=stage)
                self.T += 0.5 * p.mass * np.sum(p.velocity ** 2)
                
        self.T /= self.Number
     
    def Berendsen(self, tem):
        """Berendsen thermostat for temperature control"""
        NDIM = 2
        
        factor = np.sqrt(0.5 * NDIM * tem / self.T)
        
        self.T = 0.
        for c in self.Cells:
            for p in c.Particles:
                p.velocity *= factor
                self.T += 0.5 * p.mass * np.sum(p.velocity**2)
        self.T /= self.Number
        
    def Adjust_temp(self, tem):
        """Temperature adjustment for particles"""
        NDIM = 2  

        for c in self.Cells:
            for p in c.Particles:
                vv = p.mass * np.sum(p.velocity**2)
                factor = np.sqrt(NDIM * tem / vv)
                p.velocity *= factor
                
    def save_output(self, file):  
        """Save current state to configuration file"""
        N_obs = 0
        for ob in self.Ob:
            N_obs += len(ob)
        
        with open(file, 'w+') as f:
            HX = self.L[0, 1] - self.L[0, 0]
            HY = self.L[1, 1] - self.L[1, 0]
            HZ = self.L[2, 1] - self.L[2, 0]
            f.write(f'''Number of particles = {self.Number + len(self.Exit) + N_obs + len(self.Guide)}
                    A = 1.0 Angstrom (basic length-scale)
                    H0(1,1) = {HX} A
                    H0(1,2) = 0 A
                    H0(1,3) = 0 A
                    H0(2,1) = 0 A
                    H0(2,2) = {HY} A
                    H0(2,3) = 0 A
                    H0(3,1) = 0 A
                    H0(3,2) = 0 A
                    H0(3,3) = {HZ} A
                    entry_count = 7
                    auxiliary[0] = ID [reduced unit]
                    ''')
            
            # Write particles in a fixed order: Exits → Guides → Obstacles → Guide Agents → Agents
            # Always write sections in same order, even if empty, for consistent parsing
            
            f.write('10.000000\nAt\n')
            for e in self.Exit:  
                x, y, z = self.Normalization(e)
                f.write('{} {} {} {} {} {} {}\n'.format(x, y, z, 0, 0, 0, -1))   
            
            # Write Guide positions (always write section, even if empty)
            if len(self.Guide) > 0:
                f.write('1.000000\nFe\n')
                for g in self.Guide:
                    x, y, z = self.Normalization(g)
                    f.write('{} {} {} {} {} {} {}\n'.format(x, y, z, 0, 0, 0, -2))
            
            # Write obstacles (always write section, even if empty)
            for idx, b in enumerate(self.Ob): 
                if len(b) > 0:
                    if idx == 0:
                        f.write('1.000000\nC\n')
                    elif idx == 1:
                        f.write('1.000000\nSi\n')
                    
                    for e in b:
                        x, y, z = self.Normalization(e)
                        f.write('{} {} {} {} {} {} {}\n'.format(x, y, z, 0, 0, 0, -1))
                        
            # Write agents (always write both sections, even if empty)
            guide_agents = []
            normal_agents = []
            
            for c in self.Cells:
                for p in c.Particles:
                    if hasattr(p, 'is_guide') and p.is_guide:
                        guide_agents.append(p)
                    else:
                        normal_agents.append(p)
            
            # Write guide agents (always write section marker)
            if len(guide_agents) > 0:
                f.write('1.000000\nS\n')
                for p in guide_agents:
                    x, y, z = self.Normalization(p.position)
                    f.write('{} {} {} {} {} {} {}\n'.format(
                        x, y, z, 
                        p.velocity[0], p.velocity[1], p.velocity[2], p.ID))
            
            # Write normal agents (always write section marker)
            if len(normal_agents) > 0:
                f.write('1.000000\nBr\n')
                for p in normal_agents:
                    x, y, z = self.Normalization(p.position)
                    f.write('{} {} {} {} {} {} {}\n'.format(
                        x, y, z, 
                        p.velocity[0], p.velocity[1], p.velocity[2], p.ID))
            
    def insert_particle(self, particle):
        """Insert a particle into the appropriate cell"""
        # Check for NaN or inf in position and fix it
        if not np.all(np.isfinite(particle.position)):
            # Reset to center if position is invalid
            particle.position = np.array([
                (self.L[0, 0] + self.L[0, 1]) / 2,
                (self.L[1, 0] + self.L[1, 1]) / 2,
                (self.L[2, 0] + self.L[2, 1]) / 2
            ])
            particle.velocity = np.zeros_like(particle.velocity)
            particle.acc = np.zeros_like(particle.acc)
        
        # Clamp particle position to boundary
        eps = 1e-6
        particle.position = np.clip(
            particle.position,
            self.L[:, 0] + eps,
            self.L[:, 1] - eps
        )
        
        index = (particle.position - self.L[:, 0]) / self.d_cells
        
        index = np.clip(index, 0, self.n_cells - 1)
        
        index = index.astype('int')
        
        # Ensure index is valid
        index = np.minimum(index, self.n_cells - 1)
        
        N = int(index[2] * (self.n_cells[0] * self.n_cells[1]) + index[1] * self.n_cells[0] + index[0])
        
        N = max(0, min(N, len(self.Cells) - 1))
        self.Cells[N].add(particle)
    
    def _check_obstacle_collision(self, pos, agent_size):
        """Check if position collides with any obstacle (OPTIMIZED)"""
        for idx in range(len(self.Ob)):
            obs_type = self.Ob_type[idx] if idx < len(self.Ob_type) else 'points'
            
            if obs_type == 'rectangle':
                # Rectangle collision check - direct geometric test
                params = self.Ob_params[idx]
                center = params['center']
                half_width = params['width'] / 2
                half_height = params['height'] / 2
                
                # Find closest point on rectangle
                closest_x = np.clip(pos[0], center[0] - half_width, center[0] + half_width)
                closest_y = np.clip(pos[1], center[1] - half_height, center[1] + half_height)
                
                # Check distance to closest point
                dx = pos[0] - closest_x
                dy = pos[1] - closest_y
                dis = np.sqrt(dx*dx + dy*dy)
                
                if dis < agent_size:
                    return True
                    
            elif obs_type == 'circle':
                # Circle collision check - single center point
                if len(self.Ob[idx]) > 0:
                    center = self.Ob[idx][0]
                    dis = np.sqrt(np.sum((pos - center) ** 2))
                    if dis < (agent_size + self.Ob_size[idx]):
                        return True
            else:
                # Legacy point-based obstacles
                ob = self.Ob[idx]
                if isinstance(ob, list):
                    for p in ob:
                        dis = np.sqrt(np.sum((pos - p) ** 2))
                        if dis < (agent_size + self.Ob_size[idx]) / 2:
                            return True
                else:
                    dis = np.sqrt(np.sum((pos - ob) ** 2))
                    if dis < (agent_size + self.Ob_size[idx]) / 2:
                        return True
        
        return False
    
    def _find_valid_position(self, initial_pos, P_list, agent_size, max_attempts=50):
        """Try to find a valid position by moving away from obstacles or retrying random positions"""
        pos = initial_pos.copy()
        
        # Try the initial position first
        if not self._check_obstacle_collision(pos, agent_size):
            # Check overlap with existing agents
            overlap = False
            for p in P_list:
                dis = np.sqrt(np.sum((pos - p) ** 2))
                if dis < agent_size:
                    overlap = True
                    break
            if not overlap:
                return pos
        
        # If initial position is blocked, try moving outward in a spiral pattern
        for attempt in range(max_attempts):
            # Try moving in different directions
            angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)  # 8 directions
            distances = [0.05, 0.1, 0.15, 0.2, 0.25]  # Try different distances
            
            for dist in distances:
                for angle in angles:
                    # Calculate new position
                    offset_x = dist * (self.L[0, 1] - self.L[0, 0]) * np.cos(angle)
                    offset_y = dist * (self.L[1, 1] - self.L[1, 0]) * np.sin(angle)
                    new_pos = pos.copy()
                    new_pos[0] += offset_x
                    new_pos[1] += offset_y
                    
                    # Check if within bounds
                    if (new_pos[0] < self.L[0, 0] or new_pos[0] > self.L[0, 1] or
                        new_pos[1] < self.L[1, 0] or new_pos[1] > self.L[1, 1]):
                        continue
                    
                    # Check obstacle collision
                    if self._check_obstacle_collision(new_pos, agent_size):
                        continue
                    
                    # Check overlap with existing agents
                    overlap = False
                    for p in P_list:
                        dis = np.sqrt(np.sum((new_pos - p) ** 2))
                        if dis < agent_size:
                            overlap = True
                            break
                    
                    if not overlap:
                        return new_pos
            
            # If we've tried all nearby positions and failed, generate a completely new random position
            pos = (self.L[0:2, 0] + 0.05 * (self.L[0:2, 1] - self.L[0:2, 0]) + 
                   np.random.rand(2) * (self.L[0:2, 1] - self.L[0:2, 0]) * 0.9)
            pos = np.append(pos, self.L[2, 0] + 0.5 * (self.L[2, 1] - self.L[2, 0]))
        
        # Last resort: return the last attempted position (may still be invalid)
        return pos
    
    def initialize_particles(self, file=None):
        """Initialize particles with random positions"""
        if file is None:
            P_list = []
            for i in range(self.Number):
                # Initial positions of 2D random distribution at xy plane
                initial_pos = (self.L[0:2, 0] + 0.05 * (self.L[0:2, 1] - self.L[0:2, 0]) + 
                               np.random.rand(len(self.L) - 1) * (self.L[0:2, 1] - self.L[0:2, 0]) * 0.9)
                initial_pos = np.append(initial_pos, self.L[2, 0] + 0.5 * (self.L[2, 1] - self.L[2, 0]))
                
                # Find a valid position (move away from obstacles if needed)
                pos = self._find_valid_position(initial_pos, P_list, agent_size)
                
                P_list.append(pos)                
                pos = pos.tolist()
                
                # Set initial velocity
                v = np.random.randn(len(self.L)) * 0.01
                v[2] = 0.
                v = v.tolist()
                particle = Particle(i, *pos, *v)
                
                if i == 0:
                    self.agent = particle
                
                self.insert_particle(particle)
        else:  
            # Read from Cfg file
            pass

    def move_particles(self):
        """Check positions of agents and move them to the right cell"""
        for cell in self.Cells:
            i = 0
            while i < len(cell.Particles):
                position = cell.Particles[i].position
                
                inside = (position >= cell.L[:, 0]) & (position < cell.L[:, 1])
                inside = inside.all()
                
                if inside:
                    i += 1
                else:                
                    self.insert_particle(cell.Particles.pop(i))                                        
                    
    def _compute_obstacle_force_optimized(self, particle_pos, ob_idx):
        """Optimized obstacle force calculation with early distance check."""
        obs_type = self.Ob_type[ob_idx] if ob_idx < len(self.Ob_type) else 'points'
        
        if obs_type == 'rectangle':
            # Direct rectangle distance calculation (much faster than point grid)
            params = self.Ob_params[ob_idx]
            center = params['center']
            half_width = params['width'] / 2
            half_height = params['height'] / 2
            
            # Find closest point on rectangle to particle
            closest_x = np.clip(particle_pos[0], center[0] - half_width, center[0] + half_width)
            closest_y = np.clip(particle_pos[1], center[1] - half_height, center[1] + half_height)
            closest_z = center[2]
            
            # Distance to closest point
            dr = particle_pos - np.array([closest_x, closest_y, closest_z])
            dis = np.sqrt(np.sum(dr ** 2)) + 1e-10
            dis_eq = agent_size + 0.05  # Small collision margin for rectangles
            
            if dis < dis_eq:
                f = f_collision_lim * np.exp((dis_eq - dis) / 0.08)
                return f * dr / dis
            return np.zeros(3)
            
        elif obs_type == 'circle':
            # Circle obstacle - single point check
            if len(self.Ob[ob_idx]) > 0:
                center = self.Ob[ob_idx][0]
                dr = particle_pos - center
                dis = np.sqrt(np.sum(dr ** 2)) + 1e-10
                dis_eq = (agent_size + self.Ob_size[ob_idx]) / 2
                
                if dis < dis_eq:
                    f = f_collision_lim * np.exp((dis_eq - dis) / 0.08)
                    return f * dr / dis
            return np.zeros(3)
            
        else:
            # Legacy point-based obstacles (backward compatibility)
            # Use distance threshold to skip far obstacles
            total_force = np.zeros(3)
            max_force_distance = agent_size * 3  # Only compute forces within 3x agent size
            
            for i in self.Ob[ob_idx]:
                dr = particle_pos - i
                dis_sq = np.sum(dr ** 2)
                
                # Early rejection: skip if too far
                if dis_sq > max_force_distance ** 2:
                    continue
                    
                dis = np.sqrt(dis_sq) + 1e-10
                dis_eq = (agent_size + self.Ob_size[ob_idx]) / 2
                
                if dis < dis_eq:
                    f = f_collision_lim * np.exp((dis_eq - dis) / 0.08)
                    total_force += f * dr / dis
            
            return total_force
    
    def region_confine(self):
        """Apply region confining forces: walls, obstacles and friction (OPTIMIZED)"""
        for c in self.Cells:
            for p in c.Particles:
                # Wall forces
                dis = p.position[:, np.newaxis] - self.L    
                dis = np.abs(dis)
                f = np.where(dis < agent_size, 
                           f_wall_lim * np.exp((agent_size - dis) / 0.08) * dis, 0.) 
                f[:, 1] = -f[:, 1]
                f = f.sum(axis=1)
                
                p.acc += f / p.mass         
                
                # Obstacle forces (OPTIMIZED)
                for idx in range(len(self.Ob)):
                    f_obs = self._compute_obstacle_force_optimized(p.position, idx)
                    p.acc += f_obs / p.mass
                
                # Friction force
                f = -p.mass / relaxation_time * p.velocity
                p.acc += f / p.mass            
                
    def loop_cells(self):
        """Loop particles in the same cell to calculate collision forces"""
        for c in self.Cells:            
            l = len(c.Particles)
            
            for i in range(l):
                for j in range(i + 1, l):
                    p1 = c.Particles[i]
                    p2 = c.Particles[j]
                    
                    dr = p1.position - p2.position
                    dis = np.sqrt(np.sum(dr ** 2)) + 1e-10
                    
                    if dis < agent_size:
                        f = f_collision_lim * np.exp((agent_size - dis) / 0.08)                        
                        f = f * dr / dis
                        p1.acc += f / p1.mass
                        p2.acc -= f / p2.mass
    
    def loop_neighbors(self):
        """Loop particles in neighbor cells to calculate collision forces"""
        for c in self.Cells:               
            for n in c.Neighbors:
                for p1 in c.Particles:
                    for p2 in self.Cells[n].Particles:
                        dr = p1.position - p2.position
                        dis = np.sqrt(np.sum(dr ** 2)) + 1e-10
                        
                        if dis < agent_size:
                            f = f_collision_lim * np.exp((agent_size - dis) / 0.08)
                            f = f * dr / dis
                            p1.acc += f / p1.mass
                            p2.acc -= f / p2.mass                        
    
    def reset(self):
        """Reset initial configuration the Continuum cell space"""
        for cell in self.Cells:
            cell.Particles.clear()
            
        self.Number = self.Total
        self.initialize_particles()
        
        return (self.agent.position[0], self.agent.position[1],
                self.agent.velocity[0], self.agent.velocity[1])

    def choose_random_action(self):
        """Choose random action from action list"""
        action = np.random.choice(len(self.action))
        return action
    
    def Get_Neighbior_Cells(self, position):
        """Get the number of particles from neighbor cells"""
        position = np.array(position)
        position = np.append(position, self.L[2, 0] + 0.5 * (self.L[2, 1] - self.L[2, 0]))
        index = (position - self.L[:, 0]) / self.d_cells
        index = index.astype('int')
        
        idx = index + neighbor_list        
        valid = (idx < self.n_cells) & (idx >= 0)
        mask = np.all(valid, axis=1)   
        idx = idx[mask]
        
        neighbors = []
       
        for d in range(len(idx)):
            i = idx[d, 0]
            j = idx[d, 1]
            k = idx[d, 2]
            
            N = k * (self.n_cells[0] * self.n_cells[1]) + j * self.n_cells[0] + i            
            neighbors.append(len(self.Cells[N].Particles))
        
        neighbor_cells = np.zeros(9)
        neighbor_cells[mask] = neighbors
        
        return neighbor_cells

    def step(self, action):
        """
        Step function for agent 0 taking certain action and others taking random actions.
        Used during training of smart agents.
        """
        reward = self.reward
        done = False
        
        self.Zero_acc()
        self.update_visibility_system()  # Update exit visibility based on path congestion
 
        self.region_confine()
        self.loop_cells()
        self.loop_neighbors()
        
        for c in self.Cells:
            for p in c.Particles:
                if p.ID != 0:
                    action = np.random.choice(len(self.action))
                    
                p.acc += 1 / relaxation_time * desire_velocity * self.action[action]
        
        self.Integration(1)               
        self.Integration(0)
        self.move_particles()
        
        next_state = (self.agent.position[0], self.agent.position[1],
                      self.agent.velocity[0], self.agent.velocity[1])
        
        for e in self.Exit:
            dis = self.agent.position - e
            dis = np.sqrt(np.sum(dis ** 2))
            if dis < dis_lim:
                done = True
                reward = self.end_reward
                break
        
        return next_state, reward, done
    
    def step_all_pytorch(self, QNet, device, Normalized=False):
        """
        Step function for all agents taking optimal actions from the QNet (PyTorch version).
        This assumes all agents are smart and can make optimal decisions.
        """
        import torch
        
        done = False

        if self.Number == 0:
            done = True
            return done
        
        self.Zero_acc()
        self.update_visibility_system()  # Update exit visibility based on path congestion
 
        self.region_confine()
        self.loop_cells()
        self.loop_neighbors()
        
        for c in self.Cells:
            for p in c.Particles:
                state = np.array([p.position[0], p.position[1], p.velocity[0], p.velocity[1]])
                
                if Normalized:
                    state[:2] = self.Normalization_XY(state[:2])
                
                # Convert to PyTorch tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                # Get Q-values from network
                with torch.no_grad():
                    Qs = QNet(state_tensor).cpu().numpy()[0]
                    
                action_list = [idx for idx, val in enumerate(Qs) if val == np.max(Qs)]                    
                action = np.random.choice(action_list)

                p.acc += 1 / relaxation_time * desire_velocity * self.action[action]
        
        self.Integration(1)        
        self.Integration(0)
        self.move_particles()
          
        for c in self.Cells:
            i = 0
            while i < len(c.Particles):
                in_exit = False
                for e in self.Exit:
                    # Sphere region
                    dis = c.Particles[i].position - e
                    dis = np.sqrt(np.sum(dis ** 2))
                    if dis < dis_lim:
                        c.Particles.pop(i)
                        in_exit = True
                        self.Number -= 1
                        break
                
                if not in_exit:
                    i += 1
            
        if self.Number == 0:
            done = True
        
        return done
        
    def step_optimal(self):
        """
        Step function assuming all agents know the optimal action (nearest exit).
        This represents the "smart agents" scenario where everyone makes optimal decisions.
        """
        done = False

        if self.Number == 0:
            done = True
            return done
        
        self.Zero_acc()
        self.update_visibility_system()  # Update exit visibility based on path congestion
 
        self.region_confine()
        self.loop_cells()
        self.loop_neighbors()
        
        for c in self.Cells:
            for p in c.Particles:
                # Find nearest exit
                dr = np.inf
                dr_unit = None
                for e in self.Exit:
                    dr_tmp = np.sqrt(np.sum((e - p.position) ** 2))
                    if dr_tmp < dr:
                        dr = dr_tmp
                        dr_unit = (e - p.position) / dr_tmp                
                
                # Find action that best aligns with direction to nearest exit
                costheta = -np.inf
                for action in self.action:
                    costheta_tmp = np.matmul(action, dr_unit)
                    if costheta_tmp > costheta:
                        costheta = costheta_tmp
                        dr_action = action
 
                p.acc += 1 / relaxation_time * desire_velocity * dr_action
        
        self.Integration(1)        
        self.Integration(0)
        self.move_particles()
          
        for c in self.Cells:
            i = 0
            while i < len(c.Particles):
                in_exit = False
                for e in self.Exit:
                    dis = c.Particles[i].position - e
                    dis = np.sqrt(np.sum(dis ** 2))
                    if dis < dis_lim:
                        c.Particles.pop(i)
                        in_exit = True
                        self.Number -= 1
                        break
                
                if not in_exit:
                    i += 1
            
        if self.Number == 0:
            done = True
        
        return done


if __name__ == '__main__':
    # Test particle motions
    a = Cell_Space(0, 50, 0, 50, 0, 2, rcut=1.5, dt=delta_t, Number=2000)
    state = a.reset()
    max_time = 1000
    Cfg_path = './Cfg'
    if not os.path.isdir(Cfg_path):
        os.mkdir(Cfg_path)    
    
    cases = 1
    
    for i in range(cases):
        t = 0
        pathdir = Cfg_path + '/case_' + str(i)
        
        if not os.path.isdir(pathdir):
            os.mkdir(pathdir)
        
        a.save_output(pathdir + '/s.' + str(t))
        
        while t < max_time:   
            print("step: {}".format(t))
            action = np.random.choice(len(a.action))             
            next_state, reward, done = a.step(action)
            
            t += 1
            
            if done:
                a.save_output(pathdir + '/s.' + str(t)) 
                state = a.reset()
                break
            
            state = next_state
            
            if t % cfg_save_step == 0:
                a.save_output(pathdir + '/s.' + str(t))

