"""
Continuum Cell Space Environment for Agent (Particle) Dynamics

This module implements the physical simulation environment for emergency evacuation,
including particle dynamics, cell-based spatial partitioning, and force calculations.

Step functions:
- step(action)   [Cell_Space]       Single-agent RL: agent 0 takes `action`, others random. Returns (next_state, reward, done).
- step_guided()  [GuidedCellSpace]  Main evacuation: evacuees KNN+noise/exit-directed; guide agents collision only. Returns done. Used by run_guided_visualize.
"""

import heapq
from collections import deque
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
    
    def _remove_particles_at_exits(self):
        """
        Remove evacuee particles that have reached any exit (distance < dis_lim).
        Guide agents (is_guide=True) are never removed so they can be used for
        future RL (e.g. trainable guide that stays in the scene).
        Updates self.Number. Call after move_particles() in each step.
        """
        for c in self.Cells:
            i = 0
            while i < len(c.Particles):
                p = c.Particles[i]
                if getattr(p, 'is_guide', False):
                    i += 1
                    continue
                in_exit = False
                for e in self.Exit:
                    dis = np.sqrt(np.sum((p.position - e) ** 2))
                    if dis < dis_lim:
                        c.Particles.pop(i)
                        in_exit = True
                        self.Number -= 1
                        break
                if not in_exit:
                    i += 1

    def _correct_obstacle_penetration(self):
        """Correct particle positions that have penetrated obstacles.
        This is called after move_particles to ensure no particles remain inside obstacles.
        """
        for c in self.Cells:
            for p in c.Particles:
                penetration_depth, correction_vec = self._get_obstacle_penetration_depth(p.position, agent_size)
                
                if penetration_depth > 0:
                    # Immediately correct position by pushing particle out
                    p.position += correction_vec
                    
                    # Also apply strong velocity correction to prevent re-entry
                    # Reverse velocity component in penetration direction
                    if np.linalg.norm(correction_vec) > 1e-6:
                        correction_dir = correction_vec / np.linalg.norm(correction_vec)
                        # Project velocity onto correction direction and reverse it
                        vel_proj = np.dot(p.velocity[:2], correction_dir[:2])
                        if vel_proj < 0:  # Moving into obstacle
                            p.velocity[:2] -= 2 * vel_proj * correction_dir[:2]  # Reflect velocity
                            p.velocity[:2] *= 0.5  # Dampen velocity to prevent oscillation                                        
                    
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
                # Wall forces (same strength curve as obstacles: strong when close)
                dis = p.position[:, np.newaxis] - self.L
                dis = np.abs(dis)
                penetration = np.maximum(agent_size - dis, 0.0)
                f = np.where(dis < agent_size, f_wall_lim * np.exp(penetration / 0.05), 0.)
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


class GuidedParticle(Particle):
    """
    Extended particle for guided evacuation scenarios

    Attributes:
        knows_exit: Whether this agent knows the location of the nearest exit
        follow_threshold: Velocity threshold for following crowd behavior
        social_weight: Weight given to social following behavior
        is_guide: Whether this agent acts as a guide

        first_guided: Whether this evacuee has ever been guided by a robot
        exit_path_memory: Last remembered A* direction to the nearest exit
        memory_strength: How strongly the evacuee follows the remembered path [0, 1]
    """

    def __init__(self, ID, x, y, z, vx, vy, vz, mass=80.0, type=1,
                 knows_exit=False, follow_threshold=1.0, social_weight=0.5,
                 is_guide=False):
        super().__init__(ID, x, y, z, vx, vy, vz, mass, type)
        self.knows_exit = knows_exit
        self.follow_threshold = follow_threshold
        self.social_weight = social_weight
        self.is_guide = is_guide
        # Guide-memory related state (only used for evacuees, ignored for guides)
        # first_guided: has this evacuee ever been guided by a robot
        self.first_guided = False
        # needs_first_memory_reward: whether the "first guided" bonus is still pending
        self.needs_first_memory_reward = True
        # just_guided_this_step: internal flag to mark first guided event in current step
        self.just_guided_this_step = False
        # Store as 3D vector (z=0) for convenience when mixing with other directions
        self.exit_path_memory = np.zeros(3, dtype=np.float64)
        self.memory_strength = 0.0
        # Step index when exit_path_memory was last updated from A* (for periodic refresh)
        self.last_astar_update_step = -999


class GuidedCellSpace(Cell_Space):
    """
    Extended Cell Space for guided evacuation
    
    This environment implements:
    - Proximity-based exit knowledge
    - Crowd following behavior
    - Guide agent interactions
    """
    
    def __init__(self, xmin=0., xmax=1., ymin=0., ymax=1.,
                 zmin=0., zmax=1., rcut=0.5, dt=0.01, Number=1,
                 door_visible_radius=1.0, knn_k=5,
                 guide_radius=1, perception_radius=2.5, use_knn=True, speed_scale=1.0,
                 guide_speed_scale=None, obstacle_configs=None, knn_max_distance=3.0,
                 knn_filter_obstacles=True, n_guide_agent=0,
                 guide_initial_position_mode='random', guide_initial_position=None,
                 memory_increase_rate=5.0, memory_decay_rate=0.2,
                 memory_astar_thres_around=0.3, memory_astar_thres_out=0.2, memory_astar_update_interval_n=5):
        self.n_guide_agent = max(0, int(n_guide_agent))
        self.guide_radius = guide_radius
        self.perception_radius = float(perception_radius)
        # Separate speed scales: speed_scale for evacuees (particles), guide_speed_scale for guide agent(s)
        self.speed_scale = max(0.1, float(speed_scale))
        self.guide_speed_scale = max(0.1, float(guide_speed_scale)) if guide_speed_scale is not None else self.speed_scale
        # Memory dynamics for evacuees once guided by a robot
        self.memory_increase_rate = max(0.0, float(memory_increase_rate))
        self.memory_decay_rate = max(0.0, float(memory_decay_rate))
        self.memory_astar_thres_around = float(memory_astar_thres_around)
        self.memory_astar_thres_out = float(memory_astar_thres_out)
        self.memory_astar_update_interval_n = max(1, int(memory_astar_update_interval_n))
        self.guide_initial_position_mode = str(guide_initial_position_mode).strip().lower()
        self.guide_initial_position = np.array(guide_initial_position, dtype=float) if guide_initial_position is not None else None
        # Store obstacle configs BEFORE calling parent __init__ (which calls initialize_particles)
        self.obstacle_configs = obstacle_configs if obstacle_configs is not None else []
        super().__init__(xmin, xmax, ymin, ymax, zmin, zmax, rcut, dt, Number)
        self.n_particle_initial = self.Number  # for state normalization (evacuees in guide range)
        # door_visible_radius: distance at which agents start rushing toward exit directly
        self.door_visible_radius = door_visible_radius
        self.knn_k = max(1, int(knn_k))
        self.use_knn = use_knn
        self.knn_max_distance = knn_max_distance
        self.knn_filter_obstacles = knn_filter_obstacles

    def region_confine(self):
        """Vectorized region confining: wall + obstacle + friction over all particles (faster than per-particle loop)."""
        particles = []
        for c in self.Cells:
            for p in c.Particles:
                particles.append(p)
        if not particles:
            return
        N = len(particles)
        pos = np.array([p.position for p in particles], dtype=np.float64)
        vel = np.array([p.velocity for p in particles], dtype=np.float64)
        mass = np.array([p.mass for p in particles], dtype=np.float64)
        acc = np.zeros_like(pos)
        ag = agent_size
        # Wall forces (vectorized): same form as obstacle repulsion (no * dis_abs so force stays strong when close)
        dis = np.empty((N, 3, 2))
        dis[:, :, 0] = pos - self.L[:, 0]
        dis[:, :, 1] = self.L[:, 1] - pos
        dis_abs = np.abs(dis)
        penetration_wall = np.maximum(ag - dis_abs, 0.0)
        f = np.where(dis_abs < ag, f_wall_lim * np.exp(penetration_wall / 0.05), 0.0)
        f[:, :, 1] = -f[:, :, 1]
        acc += f.sum(axis=2) / mass[:, np.newaxis]
        # Obstacle forces (loop obstacles, vectorized over particles)
        if hasattr(self, 'obstacle_configs') and self.obstacle_configs:
            for obs_cfg in self.obstacle_configs:
                obs_type = obs_cfg.get('type', 'circle')
                if obs_type == 'circle':
                    center = np.array([obs_cfg['x'], obs_cfg['y'], obs_cfg.get('z', 0.5)], dtype=np.float64)
                    radius = float(obs_cfg.get('size', 0.5))
                    dr = pos - center
                    dis_to_center = np.sqrt(np.sum(dr[:, :2] ** 2, axis=1)) + 1e-10
                    dis_to_boundary = dis_to_center - radius
                    mask = dis_to_boundary < ag
                    force_mult = np.where(dis_to_boundary < 0, 5.0, 1.0)
                    penetration = np.maximum(ag - dis_to_boundary, 0)
                    f_mag = np.where(mask, f_collision_lim * force_mult * np.exp(penetration / 0.05), 0.0)
                    dir_2d = dr[:, :2] / (dis_to_center[:, np.newaxis] + 1e-10)
                    f_vec = np.zeros((N, 3))
                    f_vec[:, :2] = f_mag[:, np.newaxis] * dir_2d
                    acc += f_vec / mass[:, np.newaxis]
                elif obs_type == 'rectangle':
                    cx, cy = obs_cfg['x'], obs_cfg['y']
                    w, h = obs_cfg.get('width', 0.4), obs_cfg.get('height', 0.3)
                    left, right = cx - w / 2, cx + w / 2
                    bottom, top = cy - h / 2, cy + h / 2
                    closest_x = np.clip(pos[:, 0], left, right)
                    closest_y = np.clip(pos[:, 1], bottom, top)
                    dr_2d = pos[:, :2] - np.column_stack([closest_x, closest_y])
                    dis_rect = np.sqrt(np.sum(dr_2d ** 2, axis=1)) + 1e-10
                    is_inside = (pos[:, 0] >= left) & (pos[:, 0] <= right) & (pos[:, 1] >= bottom) & (pos[:, 1] <= top)
                    dist_to_edges = np.minimum(
                        np.minimum(pos[:, 0] - left, right - pos[:, 0]),
                        np.minimum(pos[:, 1] - bottom, top - pos[:, 1])
                    )
                    penetration = np.where(is_inside, ag - dist_to_edges, np.maximum(ag - dis_rect, 0))
                    mask = (dis_rect < ag) | is_inside
                    force_mult = np.where(is_inside, 5.0, 1.0)
                    f_mag = np.where(mask, f_collision_lim * force_mult * np.exp(penetration / 0.05), 0.0)
                    f_vec = np.zeros((N, 3))
                    f_vec[:, :2] = f_mag[:, np.newaxis] * (dr_2d / (dis_rect[:, np.newaxis] + 1e-10))
                    acc += f_vec / mass[:, np.newaxis]
        # Friction (vectorized)
        acc += (-mass[:, np.newaxis] / relaxation_time * vel) / mass[:, np.newaxis]
        # Penetration correction: keep per-particle (few particles need it)
        for i, p in enumerate(particles):
            penetration_depth, correction_vec = self._get_obstacle_penetration_depth(p.position, ag)
            if penetration_depth > ag * 0.5:
                acc[i] += correction_vec * f_collision_lim * 10.0 / p.mass
        for i, p in enumerate(particles):
            p.acc += acc[i]

    def _loop_cells_vectorized(self):
        """Vectorized pairwise forces within each cell (matrix ops instead of Python double loop)."""
        ag = agent_size
        for c in self.Cells:
            parts = c.Particles
            L = len(parts)
            if L < 2:
                continue
            pos = np.array([p.position for p in parts], dtype=np.float64)
            mass = np.array([p.mass for p in parts], dtype=np.float64)
            dr = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dis = np.sqrt(np.sum(dr ** 2, axis=2)) + 1e-10
            np.fill_diagonal(dis, np.inf)
            mask = dis < ag
            f_mag = np.where(mask, f_collision_lim * np.exp((ag - dis) / 0.08), 0.0)
            F = f_mag[:, :, np.newaxis] * dr / (dis[:, :, np.newaxis] + 1e-10)
            F_net = F.sum(axis=1)
            for i, p in enumerate(parts):
                p.acc += F_net[i] / p.mass

    def _loop_neighbors_vectorized(self):
        """Vectorized pairwise forces between each cell and its neighbor cells."""
        ag = agent_size
        for c in self.Cells:
            p1_list = c.Particles
            for n in c.Neighbors:
                p2_list = self.Cells[n].Particles
                if not p1_list or not p2_list:
                    continue
                pos1 = np.array([p.position for p in p1_list], dtype=np.float64)
                pos2 = np.array([p.position for p in p2_list], dtype=np.float64)
                mass1 = np.array([p.mass for p in p1_list], dtype=np.float64)
                mass2 = np.array([p.mass for p in p2_list], dtype=np.float64)
                L1, L2 = len(p1_list), len(p2_list)
                dr = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]
                dis = np.sqrt(np.sum(dr ** 2, axis=2)) + 1e-10
                mask = dis < ag
                f_mag = np.where(mask, f_collision_lim * np.exp((ag - dis) / 0.08), 0.0)
                F = f_mag[:, :, np.newaxis] * dr / (dis[:, :, np.newaxis] + 1e-10)
                F_net1 = F.sum(axis=1)
                F_net2 = -F.sum(axis=0)
                for i, p in enumerate(p1_list):
                    p.acc += F_net1[i] / p.mass
                for j, p in enumerate(p2_list):
                    p.acc += F_net2[j] / p.mass

    def loop_cells(self):
        """Override: use vectorized pairwise forces within cells."""
        self._loop_cells_vectorized()

    def loop_neighbors(self):
        """Override: use vectorized pairwise forces between neighbor cells."""
        self._loop_neighbors_vectorized()

    def Integration(self, stage):
        """Override: vectorized leapfrog over all particles (avoids Python per-particle loop)."""
        particles = []
        for c in self.Cells:
            for p in c.Particles:
                particles.append(p)
        if not particles:
            self.T = 0.0
            return
        pos = np.array([p.position for p in particles], dtype=np.float64)
        vel = np.array([p.velocity for p in particles], dtype=np.float64)
        acc = np.array([p.acc for p in particles], dtype=np.float64)
        mass = np.array([p.mass for p in particles], dtype=np.float64)
        dt = self.dt
        if stage == 0:
            vel += dt / 2 * acc
            pos += dt * vel
        else:
            vel = vel + dt / 2 * acc
        for i, p in enumerate(particles):
            p.velocity = vel[i]
            p.position = pos[i]
        self.T = float(0.5 * np.sum(mass * np.sum(vel ** 2, axis=1)) / len(particles))
    
    def _check_obstacle_collision(self, pos, agent_size):
        """Check if position collides with any obstacle (using true geometric shapes)
        
        Returns True if particle center is inside or too close to obstacle boundary.
        For proper collision detection, we check if the particle's center is within
        (obstacle_boundary + agent_size) distance.
        """
        # All coordinates are now in absolute format
        
        # Check against original obstacle configurations if available
        if hasattr(self, 'obstacle_configs') and self.obstacle_configs:
            for obs_cfg in self.obstacle_configs:
                obs_type = obs_cfg.get('type', 'circle')
                
                if obs_type == 'circle':
                    # Check if particle center is too close to circle boundary
                    center = np.array([obs_cfg['x'], obs_cfg['y'], obs_cfg.get('z', 0.5)])
                    radius = obs_cfg.get('size', 0.5)  # Radius in absolute units
                    
                    dis_to_center = np.sqrt(np.sum((pos[:2] - center[:2]) ** 2))
                    # Collision if particle center is within (radius + agent_size) of circle center
                    if dis_to_center < radius + agent_size:
                        return True
                
                elif obs_type == 'rectangle':
                    # Check if particle center is too close to rectangle boundary
                    center_x = obs_cfg['x']
                    center_y = obs_cfg['y']
                    width = obs_cfg.get('width', 0.4)
                    height = obs_cfg.get('height', 0.3)
                    
                    # Rectangle boundaries (absolute coordinates)
                    left = center_x - width/2
                    right = center_x + width/2
                    bottom = center_y - height/2
                    top = center_y + height/2
                    
                    # Check if particle center is inside expanded rectangle (boundary + agent_size)
                    # This ensures particle cannot penetrate the rectangle
                    if (left - agent_size < pos[0] < right + agent_size and 
                        bottom - agent_size < pos[1] < top + agent_size):
                        return True
        
        # Fallback: check discrete obstacle points
        for idx, ob in enumerate(self.Ob):
            if isinstance(ob, list):
                # Multiple points (e.g., rectangle grid)
                for p in ob:
                    dis = np.sqrt(np.sum((pos - p) ** 2))
                    if dis < (agent_size + self.Ob_size[idx]) / 2:
                        return True
            else:
                # Single point (e.g., circle center)
                dis = np.sqrt(np.sum((pos - ob) ** 2))
                if dis < (agent_size + self.Ob_size[idx]) / 2:
                    return True
        
        return False
    
    def _get_obstacle_penetration_depth(self, pos, agent_size):
        """Calculate how deep a particle has penetrated into an obstacle.
        
        Returns:
            (penetration_depth, correction_vector): 
            - penetration_depth: how far inside the obstacle (0 if outside)
            - correction_vector: direction and magnitude to push particle out
        """
        max_penetration = 0.0
        correction_vec = np.zeros(3)
        
        if hasattr(self, 'obstacle_configs') and self.obstacle_configs:
            for obs_cfg in self.obstacle_configs:
                obs_type = obs_cfg.get('type', 'circle')
                
                if obs_type == 'circle':
                    center = np.array([obs_cfg['x'], obs_cfg['y'], obs_cfg.get('z', 0.5)])
                    radius = obs_cfg.get('size', 0.5)
                    
                    dr_to_center = pos - center
                    dis_to_center = np.sqrt(np.sum(dr_to_center[:2] ** 2)) + 1e-10
                    dis_to_boundary = dis_to_center - radius
                    
                    # If particle is inside or too close
                    if dis_to_boundary < agent_size:
                        penetration = agent_size - dis_to_boundary
                        if penetration > max_penetration:
                            max_penetration = penetration
                            # Push away from center
                            if dis_to_center > 1e-6:
                                direction = dr_to_center[:2] / dis_to_center
                                correction_vec[:2] = direction * (penetration + 0.1)  # Extra margin
                
                elif obs_type == 'rectangle':
                    center_x = obs_cfg['x']
                    center_y = obs_cfg['y']
                    width = obs_cfg.get('width', 0.4)
                    height = obs_cfg.get('height', 0.3)
                    
                    left = center_x - width/2
                    right = center_x + width/2
                    bottom = center_y - height/2
                    top = center_y + height/2
                    
                    # Check if inside rectangle
                    if left <= pos[0] <= right and bottom <= pos[1] <= top:
                        # Calculate penetration depth and correction direction
                        # Find closest edge
                        dist_to_left = pos[0] - left
                        dist_to_right = right - pos[0]
                        dist_to_bottom = pos[1] - bottom
                        dist_to_top = top - pos[1]
                        
                        min_dist = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
                        penetration = agent_size - min_dist + 0.1  # Extra margin
                        
                        if penetration > max_penetration:
                            max_penetration = penetration
                            # Push toward closest edge
                            if min_dist == dist_to_left:
                                correction_vec[0] = -(penetration + 0.1)
                            elif min_dist == dist_to_right:
                                correction_vec[0] = penetration + 0.1
                            elif min_dist == dist_to_bottom:
                                correction_vec[1] = -(penetration + 0.1)
                            else:  # dist_to_top
                                correction_vec[1] = penetration + 0.1
                    else:
                        # Check if too close to boundary
                        closest_x = np.clip(pos[0], left, right)
                        closest_y = np.clip(pos[1], bottom, top)
                        closest = np.array([closest_x, closest_y])
                        
                        dr = pos[:2] - closest
                        dis = np.sqrt(np.sum(dr ** 2)) + 1e-10
                        
                        if dis < agent_size:
                            penetration = agent_size - dis
                            if penetration > max_penetration:
                                max_penetration = penetration
                                correction_vec[:2] = (dr / dis) * (penetration + 0.1)
        
        return max_penetration, correction_vec
    
    def _compute_obstacle_force(self, particle_pos, agent_size):
        """Compute repulsive force from obstacles using geometric shapes.
        
        Enhanced version with stronger forces for particles inside obstacles.
        """
        total_force = np.zeros(3)
        
        # All coordinates are now in absolute format
        
        # Use geometric shapes if obstacle configs are available
        if hasattr(self, 'obstacle_configs') and self.obstacle_configs:
            for obs_cfg in self.obstacle_configs:
                obs_type = obs_cfg.get('type', 'circle')
                
                if obs_type == 'circle':
                    # Circle obstacle - coordinates and size are in absolute units
                    center = np.array([obs_cfg['x'], obs_cfg['y'], obs_cfg.get('z', 0.5)])
                    radius = obs_cfg.get('size', 0.5)  # Radius in absolute units
                    
                    # Distance from particle to circle CENTER
                    dr_to_center = particle_pos - center
                    dis_to_center = np.sqrt(np.sum(dr_to_center[:2] ** 2)) + 1e-10
                    
                    # Distance from particle to circle BOUNDARY (negative if inside)
                    dis_to_boundary = dis_to_center - radius
                    
                    # Apply force if agent is close to or inside the circle
                    if dis_to_boundary < agent_size:
                        # dis_eq is the equilibrium distance (agent should stay this far from boundary)
                        dis_eq = agent_size
                        penetration = dis_eq - dis_to_boundary
                        
                        # Stronger force multiplier if particle is inside obstacle
                        force_multiplier = 5.0 if dis_to_boundary < 0 else 1.0
                        
                        # Apply repulsive force (direction: away from center)
                        # Use exponential force that increases dramatically with penetration
                        f = f_collision_lim * force_multiplier * np.exp(penetration / 0.05)  # Smaller decay = stronger force
                        direction_2d = dr_to_center[:2] / dis_to_center
                        f_vec = np.zeros(3)
                        f_vec[:2] = f * direction_2d
                        total_force += f_vec
                
                elif obs_type == 'rectangle':
                    # Rectangle obstacle - find closest point on rectangle surface
                    center_x = obs_cfg['x']
                    center_y = obs_cfg['y']
                    width = obs_cfg.get('width', 0.4)
                    height = obs_cfg.get('height', 0.3)
                    
                    # Rectangle boundaries in absolute space
                    left = center_x - width/2
                    right = center_x + width/2
                    bottom = center_y - height/2
                    top = center_y + height/2
                    
                    # Check if particle is inside rectangle
                    is_inside = (left <= particle_pos[0] <= right and 
                                bottom <= particle_pos[1] <= top)
                    
                    # Find closest point on rectangle SURFACE to the particle
                    closest_x = np.clip(particle_pos[0], left, right)
                    closest_y = np.clip(particle_pos[1], bottom, top)
                    closest = np.array([closest_x, closest_y])
                    
                    # Distance from particle to closest point on rectangle
                    dr_2d = particle_pos[:2] - closest
                    dis = np.sqrt(np.sum(dr_2d ** 2)) + 1e-10
                    
                    # Apply force if close to or inside rectangle
                    dis_eq = agent_size
                    
                    if dis < dis_eq or is_inside:
                        # Calculate penetration depth
                        if is_inside:
                            # Particle is inside - find distance to nearest edge
                            dist_to_left = particle_pos[0] - left
                            dist_to_right = right - particle_pos[0]
                            dist_to_bottom = particle_pos[1] - bottom
                            dist_to_top = top - particle_pos[1]
                            min_dist = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
                            penetration = agent_size - min_dist
                            
                            # Determine push direction based on closest edge
                            if min_dist == dist_to_left:
                                push_dir = np.array([-1.0, 0.0])
                            elif min_dist == dist_to_right:
                                push_dir = np.array([1.0, 0.0])
                            elif min_dist == dist_to_bottom:
                                push_dir = np.array([0.0, -1.0])
                            else:  # dist_to_top
                                push_dir = np.array([0.0, 1.0])
                        else:
                            # Particle is outside but close
                            penetration = dis_eq - dis
                            push_dir = dr_2d / dis
                        
                        # Stronger force if inside
                        force_multiplier = 5.0 if is_inside else 1.0
                        
                        # Apply repulsive force
                        f = f_collision_lim * force_multiplier * np.exp(penetration / 0.05)
                        f_vec = np.zeros(3)
                        f_vec[:2] = f * push_dir
                        total_force += f_vec
        
        # Fallback: use discrete points method
        else:
            for idx, ob in enumerate(self.Ob):
                if isinstance(ob, list):
                    for i in ob:
                        dr = particle_pos - i
                        dis = np.sqrt(np.sum(dr ** 2)) + 1e-10
                        dis_eq = (agent_size + self.Ob_size[idx]) / 2
                        
                        if dis < dis_eq:
                            penetration = dis_eq - dis
                            force_multiplier = 5.0 if dis < self.Ob_size[idx] else 1.0
                            f = f_collision_lim * force_multiplier * np.exp(penetration / 0.05)
                            f_vec = f * dr / dis
                            total_force += f_vec
                else:
                    dr = particle_pos - ob
                    dis = np.sqrt(np.sum(dr ** 2)) + 1e-10
                    dis_eq = (agent_size + self.Ob_size[idx]) / 2
                    
                    if dis < dis_eq:
                        penetration = dis_eq - dis
                        force_multiplier = 5.0 if dis < self.Ob_size[idx] else 1.0
                        f = f_collision_lim * force_multiplier * np.exp(penetration / 0.05)
                        f_vec = f * dr / dis
                        total_force += f_vec
        
        return total_force
    
    def apply_wall_forces_to_particle(self, p):
        """Apply wall (domain boundary) repulsive forces to a single particle. Modifies p.acc in-place.
        Same strength curve as obstacles (no * dis so force stays strong when close to wall)."""
        dis = p.position[:, np.newaxis] - self.L
        dis = np.abs(dis)
        penetration = np.maximum(agent_size - dis, 0.0)
        f = np.where(dis < agent_size, f_wall_lim * np.exp(penetration / 0.05), 0.)
        f[:, 1] = -f[:, 1]
        f = f.sum(axis=1)
        p.acc += f / p.mass

    def apply_obstacle_forces_to_particle(self, p):
        """Apply obstacle repulsive forces to a single particle. Modifies p.acc in-place."""
        obstacle_force = self._compute_obstacle_force(p.position, agent_size)
        p.acc += obstacle_force / p.mass

    def apply_penetration_correction_force_to_particle(self, p, threshold_fraction=0.5):
        """If particle is deeply inside an obstacle, add extra correction force. Modifies p.acc in-place."""
        penetration_depth, correction_vec = self._get_obstacle_penetration_depth(p.position, agent_size)
        if penetration_depth > agent_size * threshold_fraction:
            correction_force = correction_vec * f_collision_lim * 10.0 / p.mass
            p.acc += correction_force

    def apply_friction_to_particle(self, p):
        """Apply friction (velocity damping) to a single particle. Modifies p.acc in-place."""
        f = -p.mass / relaxation_time * p.velocity
        p.acc += f / p.mass

    def apply_collision_forces_to_particle(self, p, deep_penetration_threshold=0.5):
        """
        Apply all collision-related forces to a single particle: walls, obstacles,
        optional deep-penetration correction, and friction.
        Use this for e.g. guide agents that only need collision logic (no movement policy).
        Modifies p.acc in-place.
        """
        self.apply_wall_forces_to_particle(p)
        self.apply_obstacle_forces_to_particle(p)
        self.apply_penetration_correction_force_to_particle(p, threshold_fraction=deep_penetration_threshold)
        self.apply_friction_to_particle(p)

    def region_confine(self):
        """Apply region confining forces: walls, obstacles, and friction (via modular collision helpers)."""
        for c in self.Cells:
            for p in c.Particles:
                self.apply_collision_forces_to_particle(p)
    
    def _find_valid_position(self, initial_pos, existing_positions, agent_size, max_attempts=50):
        """Try to find a valid position by moving away from obstacles or retrying random positions"""
        pos = initial_pos.copy()
        
        # Try the initial position first
        if not self._check_obstacle_collision(pos, agent_size):
            # Check overlap with existing positions
            overlap = False
            for p in existing_positions:
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
                    
                    # Check overlap with existing positions
                    overlap = False
                    for p in existing_positions:
                        dis = np.sqrt(np.sum((new_pos - p) ** 2))
                        if dis < agent_size:
                            overlap = True
                            break
                    
                    if not overlap:
                        return new_pos
            
            # If we've tried all nearby positions and failed, generate a completely new random position
            padding = 0.05
            pos = np.array([
                self.L[0, 0] + padding * (self.L[0, 1] - self.L[0, 0]) + np.random.rand() * (self.L[0, 1] - self.L[0, 0]) * (1 - 2 * padding),
                self.L[1, 0] + padding * (self.L[1, 1] - self.L[1, 0]) + np.random.rand() * (self.L[1, 1] - self.L[1, 0]) * (1 - 2 * padding),
                self.L[2, 0] + 0.5 * (self.L[2, 1] - self.L[2, 0])
            ])
        
        # Last resort: return the last attempted position (may still be invalid)
        return pos
    
    def initialize_particles(self, file=None, quiet=False):
        """Initialize guided particles with optional guide agents. If quiet=True, skip print (e.g. for reset_guided)."""
        if not quiet:
            print("Initializing particles.")
        if file is None:
            P_list = []
            # Padding around entire scene - use small padding
            padding = 0.05  # 5% of domain
            
            # Valid generation area (excluding padding)
            Lx = self.L[0, 1] - self.L[0, 0] - 2 * padding
            Ly = self.L[1, 1] - self.L[1, 0] - 2 * padding
            
            target_count = max(1, int(self.Number))
            aspect = Lx / Ly if Ly > 0 else 1.0
            nx = max(1, int(np.ceil(np.sqrt(target_count * aspect))))
            ny = max(1, int(np.ceil(target_count / nx)))
            cell_w = Lx / nx
            cell_h = Ly / ny

            positions = []
            for cell_idx in range(nx * ny):
                if len(positions) >= target_count:
                    break

                i = cell_idx % nx
                j = cell_idx // nx
                if j >= ny:
                    break

                # Grid boundaries (within padded area)
                x_min = self.L[0, 0] + padding + i * cell_w
                x_max = x_min + cell_w
                y_min = self.L[1, 0] + padding + j * cell_h
                y_max = y_min + cell_h
                z = self.L[2, 0] + 0.5 * (self.L[2, 1] - self.L[2, 0])

                # Random sampling within grid cell
                eps = 1e-6
                x = np.random.uniform(x_min + eps, x_max - eps) if x_max > x_min + 2*eps else 0.5 * (x_min + x_max)
                y = np.random.uniform(y_min + eps, y_max - eps) if y_max > y_min + 2*eps else 0.5 * (y_min + y_max)

                initial_pos = np.array([x, y, z])
                
                # Find a valid position (move away from obstacles if needed)
                pos = self._find_valid_position(initial_pos, positions, agent_size)
                
                # Verify position is valid before adding
                if not self._check_obstacle_collision(pos, agent_size):
                    positions.append(pos)
            
            if len(positions) < self.Number:
                self.Number = len(positions)
                self.Total = len(positions)

            # If guide(s) use fixed initial position, overwrite first n_guide positions
            n_guide = min(self.n_guide_agent, self.Number)
            if n_guide >= 1 and self.guide_initial_position_mode == 'fixed' and self.guide_initial_position is not None:
                fixed = np.asarray(self.guide_initial_position, dtype=float)
                if fixed.size >= 3:
                    # Clamp to domain
                    fixed[0] = np.clip(fixed[0], self.L[0, 0], self.L[0, 1])
                    fixed[1] = np.clip(fixed[1], self.L[1, 0], self.L[1, 1])
                    fixed[2] = np.clip(fixed[2], self.L[2, 0], self.L[2, 1])
                    positions[0] = fixed.copy()

            inserted_count = 0
            for i in range(self.Number):
                pos = positions[i]
                P_list.append(pos)

                # Initialize velocity based on proximity to exits
                # If near an exit, point toward it; otherwise random motion
                near_exit_distance = 1.5  # Distance threshold for "near exit"
                v = np.array([0., 0., 0.])
                
                nearest_exit = None
                nearest_exit_dist = np.inf
                for e in self.Exit:
                    dist = np.linalg.norm(pos - e)
                    if dist < nearest_exit_dist:
                        nearest_exit_dist = dist
                        nearest_exit = e
                
                if nearest_exit is not None and nearest_exit_dist < near_exit_distance:
                    # Near exit: point initial velocity toward the exit
                    exit_direction = nearest_exit - pos
                    exit_direction[2] = 0  # Keep z=0
                    exit_dir_norm = np.linalg.norm(exit_direction)
                    if exit_dir_norm > 1e-6:
                        v = (exit_direction / exit_dir_norm) * 0.2  # Initial speed toward exit
                    else:
                        v = np.random.randn(3) * 0.01  # Fallback to random if too close
                else:
                    # Far from exit: small random motion
                    v = np.random.randn(3) * 0.01
                
                v[2] = 0.  # Maintain z=0
                v = v.tolist()
                
                n_guide = min(self.n_guide_agent, self.Number)
                is_guide = i < n_guide
                particle = GuidedParticle(i, *pos.tolist(), *v, is_guide=is_guide)

                if i == 0:
                    self.agent = particle

                self.insert_particle(particle)
                inserted_count += 1
        else:
            super().initialize_particles(file=file)
        if not quiet:
            print(f"The number of agents is {self.Number}.")

    def reset_guided(self, quiet=True):
        """Reset the guided environment for a new episode."""
        # Reset progress tracking for new episode
        self._prev_evacuee_distances_in_range = {}
        """
        Reset environment for a new episode: clear particles and re-initialize (evacuees + guide).
        Reuses the same grid/obstacles/A* cache. Use this in training to avoid setup_environment each episode.
        """
        for c in self.Cells:
            c.Particles.clear()
        self.Number = self.n_particle_initial
        self.Total = self.n_particle_initial
        self.initialize_particles(file=None, quiet=quiet)
        # Clear go_find target cache so new episode gets fresh target
        if hasattr(self, '_go_find_steps_remaining'):
            self._go_find_steps_remaining = 0
        if hasattr(self, '_go_find_target_xy'):
            self._go_find_target_xy = None

    def _line_intersects_circle(self, p1, p2, center, radius):
        """
        Check if line segment from p1 to p2 intersects with a circle.
        
        Args:
            p1, p2: Start and end points of line segment (2D in x-y plane)
            center: Circle center (2D in x-y plane)
            radius: Circle radius
            
        Returns:
            True if the line segment intersects the circle
        """
        # Convert to 2D if needed
        p1_2d = p1[:2]
        p2_2d = p2[:2]
        center_2d = center[:2]
        
        # Vector from p1 to p2
        dx = p2_2d[0] - p1_2d[0]
        dy = p2_2d[1] - p1_2d[1]
        
        # Vector from p1 to circle center
        fx = p1_2d[0] - center_2d[0]
        fy = p1_2d[1] - center_2d[1]
        
        # Quadratic equation coefficients for closest point on line
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - radius * radius
        
        if a < 1e-10:  # p1 and p2 are essentially the same
            return c < 0
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return False
        
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        
        # Check if intersection is within line segment
        return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)
    
    def _line_intersects_rectangle(self, p1, p2, center, width, height):
        """
        Check if line segment from p1 to p2 intersects with an axis-aligned rectangle.
        
        Args:
            p1, p2: Start and end points of line segment (2D)
            center: Rectangle center (2D)
            width: Rectangle width in x direction
            height: Rectangle height in y direction
            
        Returns:
            True if the line segment intersects the rectangle
        """
        p1_2d = p1[:2]
        p2_2d = p2[:2]
        center_2d = center[:2]
        
        # Rectangle bounds
        x_min = center_2d[0] - width / 2
        x_max = center_2d[0] + width / 2
        y_min = center_2d[1] - height / 2
        y_max = center_2d[1] + height / 2
        
        # Check if line segment intersects rectangle using parametric form
        dx = p2_2d[0] - p1_2d[0]
        dy = p2_2d[1] - p1_2d[1]
        
        # Find parameter range where line is within rectangle
        t_min = 0
        t_max = 1
        
        # Check x bounds
        if abs(dx) > 1e-10:
            t1 = (x_min - p1_2d[0]) / dx
            t2 = (x_max - p1_2d[0]) / dx
            if dx < 0:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
        else:
            if p1_2d[0] < x_min or p1_2d[0] > x_max:
                return False
        
        # Check y bounds
        if abs(dy) > 1e-10:
            t1 = (y_min - p1_2d[1]) / dy
            t2 = (y_max - p1_2d[1]) / dy
            if dy < 0:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
        else:
            if p1_2d[1] < y_min or p1_2d[1] > y_max:
                return False
        
        return t_min <= t_max
    
    def _is_line_of_sight_blocked(self, pos1, pos2):
        """
        Check if line of sight between two positions is blocked by obstacles.
        
        Args:
            pos1, pos2: Two 3D positions
            
        Returns:
            True if line of sight is blocked by obstacles
        """
        # Check against obstacle configurations (all coordinates now absolute)
        if hasattr(self, 'obstacle_configs') and self.obstacle_configs:
            for obs_cfg in self.obstacle_configs:
                obs_type = obs_cfg.get('type', 'circle')
                center = np.array([obs_cfg['x'], obs_cfg['y']])
                
                if obs_type == 'circle':
                    # Coordinates are now in absolute format
                    radius = obs_cfg.get('size', 0.5)
                    
                    if self._line_intersects_circle(pos1, pos2, center, radius):
                        return True
                
                elif obs_type == 'rectangle':
                    # All coordinates are in absolute format
                    width = obs_cfg.get('width', 0.4)
                    height = obs_cfg.get('height', 0.3)
                    
                    if self._line_intersects_rectangle(pos1, pos2, center, width, height):
                        return True
        
        # Also check against Ob (discretized obstacles)
        if hasattr(self, 'Ob') and self.Ob:
            for idx, ob in enumerate(self.Ob):
                for ob_point in ob:
                    # Check if line passes near obstacle point
                    dist_to_segment = self._point_to_line_distance(ob_point, pos1, pos2)
                    ob_radius = self.Ob_size[idx] / 2 if idx < len(self.Ob_size) else 0.1
                    
                    if dist_to_segment < ob_radius + agent_size:
                        return True
        
        return False
    
    def _point_to_line_distance(self, point, line_start, line_end):
        """Calculate minimum distance from point to line segment."""
        point_2d = point[:2]
        start_2d = line_start[:2]
        end_2d = line_end[:2]
        
        # Vector from start to end
        dx = end_2d[0] - start_2d[0]
        dy = end_2d[1] - start_2d[1]
        
        # Length squared
        len_sq = dx * dx + dy * dy
        
        if len_sq < 1e-10:  # Line segment is a point
            return np.sqrt((point_2d[0] - start_2d[0])**2 + (point_2d[1] - start_2d[1])**2)
        
        # Parameter t of closest point on line
        t = max(0, min(1, ((point_2d[0] - start_2d[0]) * dx + (point_2d[1] - start_2d[1]) * dy) / len_sq))
        
        # Closest point
        closest_x = start_2d[0] + t * dx
        closest_y = start_2d[1] + t * dy
        
        # Distance
        return np.sqrt((point_2d[0] - closest_x)**2 + (point_2d[1] - closest_y)**2)
    
    def update_exit_knowledge(self):
        """Update which agents know about exits based on proximity"""
        for c in self.Cells:
            for p in c.Particles:
                # Check if agent is close enough to any exit to know about it
                for e in self.Exit:
                    dis = np.sqrt(np.sum((p.position - e) ** 2))
                    if dis < self.door_visible_radius:
                        if hasattr(p, 'knows_exit'):
                            p.knows_exit = True
                        break
    
    def get_crowd_direction(self, particle, neighbor_radius=2.0):
        """
        Get the average movement direction of nearby agents
        
        Args:
            particle: The particle to check neighbors for
            neighbor_radius: Radius to check for neighbors
            
        Returns:
            Average velocity direction of neighbors, or None if no neighbors
        """
        # Find all particles within neighbor_radius
        neighbor_velocities = []
        
        for c in self.Cells:
            for p in c.Particles:
                if p.ID == particle.ID:
                    continue
                    
                dis = np.sqrt(np.sum((p.position - particle.position) ** 2))
                if dis < neighbor_radius:
                    # Only count neighbors moving above threshold
                    v_mag = np.sqrt(np.sum(p.velocity ** 2))
                    if hasattr(p, 'follow_threshold'):
                        threshold = p.follow_threshold
                    else:
                        threshold = 1.0
                        
                    if v_mag > threshold:
                        neighbor_velocities.append(p.velocity)
        
        if len(neighbor_velocities) == 0:
            return None
        
        # Return average direction
        avg_velocity = np.mean(neighbor_velocities, axis=0)
        return avg_velocity

    def _nearest_exit_vector(self, particle):
        nearest_exit = None
        nearest_dist = np.inf

        for e in self.Exit:
            dist = np.sqrt(np.sum((e - particle.position) ** 2))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_exit = e

        if nearest_exit is None:
            return None, np.inf

        direction = nearest_exit - particle.position
        return direction, nearest_dist

    def _get_astar_grid(self):
        """Build and cache 2D occupancy grid for A* (blocked = True). Lazy init."""
        if getattr(self, '_astar_grid', None) is not None:
            return
        cell_size = 0.25  # Grid resolution
        xmin, xmax = float(self.L[0, 0]), float(self.L[0, 1])
        ymin, ymax = float(self.L[1, 0]), float(self.L[1, 1])
        margin = getattr(self, 'agent_size', 0.2)
        nx = max(2, int(np.ceil((xmax - xmin) / cell_size)))
        ny = max(2, int(np.ceil((ymax - ymin) / cell_size)))
        self._astar_cell_size = cell_size
        self._astar_origin = (xmin, ymin)
        self._astar_nx, self._astar_ny = nx, ny
        grid = np.zeros((nx, ny), dtype=bool)
        for i in range(nx):
            for j in range(ny):
                x = xmin + (i + 0.5) * cell_size
                y = ymin + (j + 0.5) * cell_size
                pos = np.array([x, y, 0.5 * (self.L[2, 0] + self.L[2, 1])])
                if self._check_obstacle_collision(pos, margin):
                    grid[i, j] = True
        self._astar_grid = grid
        self._build_astar_direction_grids()

    def _build_astar_direction_grids(self):
        """Precompute per-cell direction and path distance to each exit (Dijkstra from each exit)."""
        if getattr(self, '_astar_direction_grids', None) is not None:
            return
        self._get_astar_grid()
        nx, ny = self._astar_nx, self._astar_ny
        cell_size = self._astar_cell_size
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        self._astar_direction_grids = []   # list of (dir_grid (nx,ny,2), valid (nx,ny))
        self._astar_distance_grids = []   # list of (nx, ny) path length in world units
        for exit_idx in range(len(self.Exit)):
            gi, gj = self._world_to_cell(self.Exit[exit_idx][0], self.Exit[exit_idx][1])
            dir_grid = np.zeros((nx, ny, 2), dtype=np.float64)
            dist_grid = np.full((nx, ny), np.inf, dtype=np.float64)
            valid = np.zeros((nx, ny), dtype=bool)
            dist_grid[gi, gj] = 0.0
            valid[gi, gj] = True
            dir_grid[gi, gj, 0], dir_grid[gi, gj, 1] = 0.0, 0.0
            heap = [(0.0, gi, gj)]
            while heap:
                cur_dist, i, j = heapq.heappop(heap)
                if cur_dist > dist_grid[i, j]:
                    continue
                wx_i, wy_i = self._cell_to_world(i, j)
                for di, dj in deltas:
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                        continue
                    if (ni, nj) != (gi, gj) and self._astar_grid[ni, nj]:
                        continue
                    step_cost = np.sqrt(di * di + dj * dj)
                    new_dist = cur_dist + step_cost * cell_size
                    if new_dist < dist_grid[ni, nj]:
                        dist_grid[ni, nj] = new_dist
                        valid[ni, nj] = True
                        wx_n, wy_n = self._cell_to_world(ni, nj)
                        dx = wx_i - wx_n
                        dy = wy_i - wy_n
                        n = np.sqrt(dx * dx + dy * dy) + 1e-10
                        dir_grid[ni, nj, 0] = dx / n
                        dir_grid[ni, nj, 1] = dy / n
                        heapq.heappush(heap, (new_dist, ni, nj))
            self._astar_direction_grids.append((dir_grid, valid))
            self._astar_distance_grids.append(dist_grid)

    def _world_to_cell(self, x, y):
        """Convert world (x,y) to grid cell (i, j); clip to valid range."""
        xmin, ymin = self._astar_origin
        i = int((x - xmin) / self._astar_cell_size)
        j = int((y - ymin) / self._astar_cell_size)
        i = np.clip(i, 0, self._astar_nx - 1)
        j = np.clip(j, 0, self._astar_ny - 1)
        return i, j

    def _cell_to_world(self, i, j):
        xmin, ymin = self._astar_origin
        x = xmin + (i + 0.5) * self._astar_cell_size
        y = ymin + (j + 0.5) * self._astar_cell_size
        return x, y

    def _astar_path(self, start_xy, goal_xy):
        """
        A* path from start_xy (x,y) to goal_xy (x,y). Returns list of (x,y) waypoints in world coords, or None.
        Uses 8-neighborhood; goal cell is always considered reachable.
        """
        self._get_astar_grid()
        si, sj = self._world_to_cell(start_xy[0], start_xy[1])
        gi, gj = self._world_to_cell(goal_xy[0], goal_xy[1])
        goal_cell = (gi, gj)
        # 8-neighbors: (di, dj)
        deltas = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        # (f, g, (i, j), parent_key)
        g_start = 0.0
        h_start = np.sqrt((gi - si)**2 + (gj - sj)**2)
        open_set = [(g_start + h_start, g_start, (si, sj), None)]
        closed = set()
        best_g = {}
        best_g[(si, sj)] = 0.0
        parent = {}
        while open_set:
            f, g, (i, j), _ = heapq.heappop(open_set)
            if (i, j) in closed:
                continue
            closed.add((i, j))
            if (i, j) == goal_cell:
                # Reconstruct path (world coords)
                path = []
                cur = (i, j)
                while cur is not None:
                    path.append(self._cell_to_world(cur[0], cur[1]))
                    cur = parent.get(cur)
                path.reverse()
                return path
            for di, dj in deltas:
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= self._astar_nx or nj < 0 or nj >= self._astar_ny:
                    continue
                # Allow stepping to goal even if that cell is blocked
                if (ni, nj) != goal_cell and self._astar_grid[ni, nj]:
                    continue
                step_cost = np.sqrt(di*di + dj*dj)
                ng = g + step_cost
                if (ni, nj) in closed:
                    continue
                if ng >= best_g.get((ni, nj), np.inf):
                    continue
                best_g[(ni, nj)] = ng
                parent[(ni, nj)] = (i, j)
                h = np.sqrt((gi - ni)**2 + (gj - nj)**2)
                heapq.heappush(open_set, (ng + h, ng, (ni, nj), None))
        return None

    def _astar_direction_to_exit(self, particle):
        """
        Best direction toward nearest exit using precomputed A* direction grids (obstacle-aware).
        Returns 3D direction vector (same format as exit_vector), or None to use direct exit vector.
        """
        self._get_astar_grid()
        if not getattr(self, '_astar_direction_grids', None) or len(self._astar_direction_grids) == 0:
            exit_vector, _ = self._nearest_exit_vector(particle)
            return exit_vector
        i, j = self._world_to_cell(particle.position[0], particle.position[1])
        nearest_idx = None
        nearest_d = np.inf
        for idx, e in enumerate(self.Exit):
            d = np.sqrt(np.sum((e - particle.position) ** 2))
            if d < nearest_d:
                nearest_d = d
                nearest_idx = idx
        if nearest_idx is None:
            exit_vector, _ = self._nearest_exit_vector(particle)
            return exit_vector
        dir_grid, valid = self._astar_direction_grids[nearest_idx]
        if valid[i, j]:
            dx, dy = dir_grid[i, j, 0], dir_grid[i, j, 1]
            if np.sqrt(dx * dx + dy * dy) >= 1e-8:
                return np.array([dx, dy, 0.0])
        exit_vector, _ = self._nearest_exit_vector(particle)
        return exit_vector

    def _astar_distance_and_direction_from_xy(self, x, y):
        """
        Single call: A* path distance and unit direction from (x, y) to the same nearest exit.
        Returns (dist, dx, dy): dist in world units, (dx, dy) unit vector. Uses one nearest-exit
        choice and one grid lookup so distance and direction always refer to the same exit/path.
        """
        self._get_astar_grid()
        nearest_idx = min(range(len(self.Exit)), key=lambda idx: np.sqrt((self.Exit[idx][0] - x) ** 2 + (self.Exit[idx][1] - y) ** 2))
        nearest = self.Exit[nearest_idx]
        euclidean = np.sqrt((nearest[0] - x) ** 2 + (nearest[1] - y) ** 2) + 1e-10
        to_exit = np.array([nearest[0] - x, nearest[1] - y])
        dir_fallback = to_exit / euclidean
        if not getattr(self, '_astar_direction_grids', None) or len(self._astar_direction_grids) == 0:
            return float(euclidean), float(dir_fallback[0]), float(dir_fallback[1])
        i, j = self._world_to_cell(x, y)
        dir_grid, valid = self._astar_direction_grids[nearest_idx]
        dist_grid = self._astar_distance_grids[nearest_idx]
        if valid[i, j]:
            dx, dy = dir_grid[i, j, 0], dir_grid[i, j, 1]
            if np.isfinite(dist_grid[i, j]):
                dist = float(dist_grid[i, j])
            else:
                dist = float(euclidean)
            if np.sqrt(dx * dx + dy * dy) >= 1e-8:
                return dist, float(dx), float(dy)
            return dist, float(dir_fallback[0]), float(dir_fallback[1])
        return float(euclidean), float(dir_fallback[0]), float(dir_fallback[1])

    def _best_action_for_direction(self, direction):
        if direction is None:
            return None

        move_norm = np.linalg.norm(direction)
        if move_norm == 0:
            return None

        move_unit = direction / move_norm

        best_action = None
        best_costheta = -np.inf
        for action in self.action:
            costheta = np.dot(action, move_unit)
            if costheta > best_costheta:
                best_costheta = costheta
                best_action = action

        return best_action

    def _detect_wall_collision(self, particle, collision_threshold=0.3):
        """
        Detect if particle is colliding with walls or obstacles.
        
        Returns:
            collision_normal: Normal vector pointing away from obstacle, or None
        """
        # Check wall collisions (domain boundaries)
        collision_normal = None
        min_distance = collision_threshold
        
        # Check boundaries in x, y dimensions
        for dim in range(2):  # Only x, y dimensions
            # Lower boundary
            dist_to_lower = particle.position[dim] - self.L[dim, 0]
            if dist_to_lower < min_distance and dist_to_lower > 0:
                wall_normal = np.zeros(3)
                wall_normal[dim] = 1.0  # Point away from wall
                collision_normal = wall_normal
                
            # Upper boundary
            dist_to_upper = self.L[dim, 1] - particle.position[dim]
            if dist_to_upper < min_distance and dist_to_upper > 0:
                wall_normal = np.zeros(3)
                wall_normal[dim] = -1.0  # Point away from wall
                collision_normal = wall_normal
        
        # Check obstacle collisions
        for idx, ob in enumerate(self.Ob):
            for ob_point in ob:
                dr = particle.position - ob_point
                dist = np.sqrt(np.sum(dr ** 2)) + 1e-10
                dis_eq = (agent_size + self.Ob_size[idx]) / 2
                
                if dist < dis_eq + collision_threshold and dist > 0:
                    # Point away from obstacle
                    current_normal = dr / dist
                    collision_normal = current_normal
        
        return collision_normal

    def _get_collision_avoidance_direction(self, particle):
        """
        Get a direction that avoids collisions.
        If colliding with wall/obstacle, return a direction away from it.
        
        Returns:
            direction vector or None
        """
        collision_normal = self._detect_wall_collision(particle, collision_threshold=0.4)
        
        if collision_normal is not None:
            # Add randomness to collision avoidance to prevent rigid behavior
            random_component = np.random.randn(3) * 0.3
            random_component[2] = 0
            
            # Blend away from collision with randomness
            avoid_direction = 0.7 * collision_normal + 0.3 * random_component
            avoid_direction[2] = 0
            avoid_norm = np.linalg.norm(avoid_direction)
            if avoid_norm > 1e-6:
                avoid_direction = avoid_direction / avoid_norm
            return avoid_direction
        
        return None

    def _knn_direction_and_variance(self, particle, k, search_cells=None):
        """
        Get direction informed by KNN neighbors with enhanced randomness.
        If search_cells is provided, search there first; if fewer than k neighbors
        are found, fall back to searching all cells (avoids too few neighbors in sparse regions).

        KNN filtering:
        - Only considers neighbors within knn_max_distance
        - If knn_filter_obstacles is True, excludes neighbors blocked by obstacles
        """
        def collect_neighbors(cells_to_search):
            out = []
            for c in cells_to_search:
                for p in c.Particles:
                    if p.ID == particle.ID:
                        continue
                    dist = np.sqrt(np.sum((p.position - particle.position) ** 2))
                    if dist > self.knn_max_distance:
                        continue
                    if self.knn_filter_obstacles and self._is_line_of_sight_blocked(particle.position, p.position):
                        continue
                    out.append((dist, p))
            return out

        neighbors = collect_neighbors(search_cells) if search_cells is not None else collect_neighbors(self.Cells)
        # 局部格子邻居不足 k 个时，用全局搜索保证有足够邻居
        if search_cells is not None and len(neighbors) < k:
            neighbors = collect_neighbors(self.Cells)

        # Get exit direction (but use it less for uncertain agents)
        exit_vector, _ = self._nearest_exit_vector(particle)
        
        # If no neighbors, use random walk instead of exit bias
        if not neighbors:
            # Random direction for agents without neighbor guidance
            random_dir = np.random.randn(3)
            random_dir[2] = 0  # Keep z=0
            random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-6)
            # High variance for true randomness
            return random_dir, 2.0  # Large variance ~1.4 m/s std dev

        neighbors.sort(key=lambda item: item[0])
        neighbors = neighbors[:max(1, k)]

        # Get neighbor velocities for conformity
        distances = np.array([d for d, _ in neighbors])
        velocities = np.array([p.velocity for _, p in neighbors])
        weights = 1.0 / (distances + 1e-6)
        weights = weights / np.sum(weights)

        avg_velocity = np.sum(velocities * weights[:, np.newaxis], axis=0)
        avg_velocity_norm = np.linalg.norm(avg_velocity)
        
        # Add random direction component for exploration
        random_dir = np.random.randn(3)
        random_dir[2] = 0
        random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-6)
        
        # Blend: primarily follow neighbors (60%), add randomness (40%)
        # This ensures agents explore multiple paths, not just follow one direction
        if avg_velocity_norm > 1e-6:
            neighbor_direction = avg_velocity / avg_velocity_norm
        else:
            neighbor_direction = avg_velocity
        
        move_vector = 0.6 * neighbor_direction + 0.4 * random_dir

        # Calculate speed variance with enhancement for uncertainty
        speeds = np.linalg.norm(velocities[:, :2], axis=1)
        weighted_mean = np.sum(speeds * weights)
        speed_variance = np.sum(weights * (speeds - weighted_mean) ** 2)
        
        # Amplify variance to add more noise - multiply by factor to increase randomness
        enhanced_variance = speed_variance * 1.5 + 0.5  # Ensure minimum noise

        return move_vector, enhanced_variance

    def _get_guiding_guide_position(self, particle):
        """Return position of a guide within guide_radius of the particle, or None."""
        for c in self.Cells:
            for p in c.Particles:
                if not getattr(p, 'is_guide', False):
                    continue
                dist = np.sqrt(np.sum((p.position - particle.position) ** 2))
                if dist <= self.guide_radius:
                    return p.position.copy()
        return None

    def _count_evacuees_in_guide_range_raw(self, guide_pos):
        """Count evacuees within guide_radius of guide_pos. Returns integer count."""
        n = 0
        for c in self.Cells:
            for p in c.Particles:
                if getattr(p, 'is_guide', False):
                    continue
                dist = np.sqrt(np.sum((p.position - guide_pos) ** 2))
                if dist <= self.guide_radius:
                    n += 1
        return n


    def _count_evacuees_in_guide_range(self, guide_pos, max_norm=100.0):
        """Count evacuees within guide_radius of guide_pos. Returns normalized count in [0, 1] by max_norm."""
        n = self._count_evacuees_in_guide_range_raw(guide_pos)
        return min(1.0, float(n) / max(1.0, max_norm))

    def _get_evacuees_perception_state(self, guide_pos):
        """
        Within perception_radius (circular range), compute:
        - Direction from guide to average position of evacuees (unit vector, 2D).
        - Average velocity direction of evacuees (unit vector, 2D).
        Returns (dir_to_avg_pos_x, dir_to_avg_pos_y, avg_vel_dir_x, avg_vel_dir_y).
        When no evacuees in range, returns (0, 0, 0, 0).
        """
        r = self.perception_radius
        gx, gy = float(guide_pos[0]), float(guide_pos[1])
        positions = []
        velocities = []
        for c in self.Cells:
            for p in c.Particles:
                if getattr(p, 'is_guide', False):
                    continue
                dx = p.position[0] - gx
                dy = p.position[1] - gy
                dist = np.sqrt(dx * dx + dy * dy)
                if dist <= r:
                    positions.append([p.position[0], p.position[1]])
                    velocities.append([p.velocity[0], p.velocity[1]])
        if not positions:
            return np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0)
        pos_arr = np.array(positions, dtype=np.float64)
        vel_arr = np.array(velocities, dtype=np.float64)
        avg_x = float(np.mean(pos_arr[:, 0]))
        avg_y = float(np.mean(pos_arr[:, 1]))
        dx = avg_x - gx
        dy = avg_y - gy
        norm_pos = np.sqrt(dx * dx + dy * dy) + 1e-10
        dir_to_avg_x = float(dx / norm_pos)
        dir_to_avg_y = float(dy / norm_pos)
        avg_vx = float(np.mean(vel_arr[:, 0]))
        avg_vy = float(np.mean(vel_arr[:, 1]))
        norm_vel = np.sqrt(avg_vx * avg_vx + avg_vy * avg_vy) + 1e-10
        avg_vel_dir_x = float(avg_vx / norm_vel)
        avg_vel_dir_y = float(avg_vy / norm_vel)
        return (
            np.float32(dir_to_avg_x), np.float32(dir_to_avg_y),
            np.float32(avg_vel_dir_x), np.float32(avg_vel_dir_y),
        )

    def _evacuee_centroid_xy(self):
        """Return (cx, cy) centroid of all evacuees (non-guide), or (None, None) if no evacuees."""
        cx, cy, count = 0.0, 0.0, 0
        for c in self.Cells:
            for p in c.Particles:
                if getattr(p, 'is_guide', False):
                    continue
                cx += p.position[0]
                cy += p.position[1]
                count += 1
        if count == 0:
            return None, None
        return cx / count, cy / count

    def get_guide_state(self, normalize=True, n_particle_norm=100.0):
        """
        Get state for the first guide agent.
        12-dimensional: [dir_to_avg_pos_xy, avg_vel_dir_xy, astar_dir_xy, x_norm, y_norm,
                        n_remaining_norm, n_escaped_this_step_norm, n_first_guided_this_step_norm, memory_sum_norm].
        - First four: within perception_radius, direction to crowd centroid and crowd average velocity unit direction.
        - Next two: A* direction to nearest exit (to compare with crowd movement).
        - Next two: guide's normalized position in the room (x_norm, y_norm in [0,1]).
        - Last four (for critic): n_remaining/n0, n_escaped_this_step/n0, n_first_guided_this_step/n0, memory_sum/n0.
        Returns None if no guide.
        """
        guide_pos = None
        for c in self.Cells:
            for p in c.Particles:
                if getattr(p, 'is_guide', False):
                    guide_pos = p.position.copy()
                    break
            if guide_pos is not None:
                break
        if guide_pos is None:
            return None
        d1, d2, v1, v2 = self._get_evacuees_perception_state(guide_pos)
        if self.Exit and len(self.Exit) > 0:
            _, astar_dx, astar_dy = self._astar_distance_and_direction_from_xy(guide_pos[0], guide_pos[1])
            astar_dx, astar_dy = np.float32(astar_dx), np.float32(astar_dy)
        else:
            astar_dx, astar_dy = np.float32(0.0), np.float32(0.0)
        xmin, xmax = float(self.L[0, 0]), float(self.L[0, 1])
        ymin, ymax = float(self.L[1, 0]), float(self.L[1, 1])
        x_span = xmax - xmin + 1e-10
        y_span = ymax - ymin + 1e-10
        x_norm = np.float32(np.clip((guide_pos[0] - xmin) / x_span, 0.0, 1.0))
        y_norm = np.float32(np.clip((guide_pos[1] - ymin) / y_span, 0.0, 1.0))
        n0 = max(1, getattr(self, 'n_particle_initial', self.Number))
        n_remaining = 0
        n_first_guided_this_step = 0
        memory_sum = 0.0
        for c in self.Cells:
            for p in c.Particles:
                if getattr(p, 'is_guide', False):
                    continue
                n_remaining += 1
                if getattr(p, 'just_guided_this_step', False):
                    n_first_guided_this_step += 1
                memory_sum += float(getattr(p, 'memory_strength', 0.0))
        n_escaped_this_step = int(getattr(self, '_n_escaped_this_step', 0))
        n_remaining_norm = np.float32(n_remaining / n0)
        n_escaped_norm = np.float32(n_escaped_this_step / n0)
        n_first_guided_norm = np.float32(n_first_guided_this_step / n0)
        memory_sum_norm = np.float32(memory_sum / n0)
        return np.array(
            [d1, d2, v1, v2, astar_dx, astar_dy, x_norm, y_norm,
             n_remaining_norm, n_escaped_norm, n_first_guided_norm, memory_sum_norm],
            dtype=np.float32,
        )

    def get_guide_position(self):
        """Return (x, y) of the first guide agent, or None if no guide."""
        for c in self.Cells:
            for p in c.Particles:
                if getattr(p, 'is_guide', False):
                    return (float(p.position[0]), float(p.position[1]))
        return None

    def get_guide_go_find_direction(self, min_distance=3.0, max_distance=5.0, stick_steps=5):
        """
        Used when guide chooses "go find people": pick a target point and return unit direction
        toward it (A* first segment from current position if path exists, else straight). Obstacle-aware.
        When stick_steps > 0, cache the target and reuse it: each step recompute direction from
        current position to the same target, then decrement counter. Smoother than direction cache.
        Returns (dx, dy) unit vector, or (0, 0) if no guide.
        """
        pos = self.get_guide_position()
        if pos is None:
            return (0.0, 0.0)
        remaining = getattr(self, '_go_find_steps_remaining', 0)
        target = getattr(self, '_go_find_target_xy', None)
        if stick_steps > 0 and remaining > 0 and target is not None:
            # Reuse cached target: direction from current pos to target (recompute each step)
            start_xy = (pos[0], pos[1])
            goal_xy = target
            path = self._astar_path(start_xy, goal_xy)
            if path and len(path) >= 2:
                dx = path[1][0] - path[0][0]
                dy = path[1][1] - path[0][1]
            else:
                dx = goal_xy[0] - start_xy[0]
                dy = goal_xy[1] - start_xy[1]
            n = np.sqrt(dx * dx + dy * dy) + 1e-10
            self._go_find_steps_remaining = remaining - 1
            return (float(dx / n), float(dy / n))
        # Pick new target
        xmin, xmax = float(self.L[0, 0]), float(self.L[0, 1])
        ymin, ymax = float(self.L[1, 0]), float(self.L[1, 1])
        margin = 0.5
        theta = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(min_distance, max_distance)
        tx = pos[0] + dist * np.cos(theta)
        ty = pos[1] + dist * np.sin(theta)
        tx = np.clip(tx, xmin + margin, xmax - margin)
        ty = np.clip(ty, ymin + margin, ymax - margin)
        start_xy = (pos[0], pos[1])
        goal_xy = (tx, ty)
        path = self._astar_path(start_xy, goal_xy)
        if path and len(path) >= 2:
            dx = path[1][0] - path[0][0]
            dy = path[1][1] - path[0][1]
        else:
            dx = goal_xy[0] - start_xy[0]
            dy = goal_xy[1] - start_xy[1]
        n = np.sqrt(dx * dx + dy * dy) + 1e-10
        if stick_steps > 0:
            self._go_find_target_xy = (float(tx), float(ty))
            self._go_find_steps_remaining = stick_steps - 1
        return (float(dx / n), float(dy / n))

    def get_all_positions_for_vis(self):
        """Return (agents_xy, guide_agents_xy) as lists of [x, y, z] for visualization."""
        agents_xy, guide_agents_xy = [], []
        for c in self.Cells:
            for p in c.Particles:
                pos = p.position.tolist() if hasattr(p.position, 'tolist') else list(p.position)
                if getattr(p, 'is_guide', False):
                    guide_agents_xy.append(pos)
                else:
                    agents_xy.append(pos)
        return agents_xy, guide_agents_xy

    def _remove_particles_at_exits(self):
        """
        Remove evacuee particles that have reached any exit.

        For guided training, we also accumulate a per-step sum of evacuee memory_strength
        at exits (used by get_guide_memory_reward for a small exit-based reward).
        Sets self._n_escaped_this_step for critic state (number who exited this step).
        """
        # Initialize accumulator if not present
        if not hasattr(self, "_exit_memory_sum_this_step"):
            self._exit_memory_sum_this_step = 0.0
        self._n_escaped_this_step = 0
        for c in self.Cells:
            i = 0
            while i < len(c.Particles):
                p = c.Particles[i]
                if getattr(p, 'is_guide', False):
                    i += 1
                    continue
                in_exit = False
                for e in self.Exit:
                    dis = np.sqrt(np.sum((p.position - e) ** 2))
                    if dis < dis_lim:
                        # Accumulate memory_strength for exit-based reward
                        m = float(getattr(p, "memory_strength", 0.0))
                        if m > 0.0:
                            self._exit_memory_sum_this_step += m
                        c.Particles.pop(i)
                        in_exit = True
                        self.Number -= 1
                        self._n_escaped_this_step += 1
                        break
                if not in_exit:
                    i += 1

    def get_guide_memory_reward(self, step_scale=0.0, first_scale=0.0, exit_scale=0.0):
        """
        Guide reward based on evacuees' exit-path memory.

        Components:
        - Continuous memory reward: while evacuees' memory_strength > 0, give small
          reward step_scale * memory_strength each step.
        - First-guided bonus: when an evacuee is guided for the first time, give a
          larger reward first_scale * memory_strength once, then clear the flag so
          it won't be repeated.
        - Exit reward: when evacuees exit successfully, reward exit_scale * memory_strength
          using the memory_strength they had at the moment of exit.
        """
        total = 0.0
        # Continuous + first-guided reward over current evacuees
        for c in self.Cells:
            for p in c.Particles:
                if getattr(p, "is_guide", False):
                    continue
                m = float(getattr(p, "memory_strength", 0.0))
                if m > 0.0 and step_scale != 0.0:
                    total += step_scale * m
                # First-guided bonus (once)
                if getattr(p, "just_guided_this_step", False) and getattr(p, "needs_first_memory_reward", True) and first_scale != 0.0:
                    total += first_scale * m
                    # Mark that first-guided bonus has been consumed
                    p.needs_first_memory_reward = False
                    p.just_guided_this_step = False
                else:
                    # Clear per-step flag if not used
                    p.just_guided_this_step = False

        # Exit-based reward (uses accumulator from _remove_particles_at_exits)
        exit_sum = float(getattr(self, "_exit_memory_sum_this_step", 0.0))
        if exit_sum > 0.0 and exit_scale != 0.0:
            total += exit_scale * exit_sum
        # Reset accumulator for next step
        self._exit_memory_sum_this_step = 0.0
        return total

    def get_time_penalty_reward(self, time_penalty_scale=0.01):
        """
        Per-evacuee time penalty: as long as evacuees have not exited, each particle
        contributes a constant negative reward every step.
        Input time_penalty_scale is the magnitude (positive, e.g. 0.01).
        Returns -time_penalty_scale * N, where N is the number of non-guide evacuees.
        """
        if time_penalty_scale == 0.0:
            return 0.0
        scale = abs(float(time_penalty_scale))
        n = 0
        for c in self.Cells:
            for p in c.Particles:
                if getattr(p, "is_guide", False):
                    continue
                n += 1
        if n == 0:
            return 0.0
        return -scale * float(n)

    def get_guide_boundary_penalty(self, margin=0.8, penalty_scale=0.5, corner_extra_scale=0.0):
        """
        Penalty when the guide is close to domain edges (walls/corners). Discourages
        the guide from staying near boundaries and especially in corners.
        - Base: penalty_scale * (margin - dist_to_nearest_wall) when dist < margin.
        - Corner: if corner_extra_scale > 0, add extra when close to both x and y walls
          (product term so corners get a larger penalty).
        """
        guide_pos = None
        for c in self.Cells:
            for p in c.Particles:
                if getattr(p, 'is_guide', False):
                    guide_pos = p.position
                    break
            if guide_pos is not None:
                break
        if guide_pos is None:
            return 0.0
        x, y = guide_pos[0], guide_pos[1]
        xmin, xmax = float(self.L[0, 0]), float(self.L[0, 1])
        ymin, ymax = float(self.L[1, 0]), float(self.L[1, 1])
        dist_x = min(x - xmin, xmax - x)
        dist_y = min(y - ymin, ymax - y)
        dist_to_nearest_wall = min(dist_x, dist_y)
        penalty = 0.0
        if dist_to_nearest_wall < margin:
            penalty = penalty_scale * (margin - dist_to_nearest_wall)
        if corner_extra_scale > 0 and dist_x < margin and dist_y < margin:
            penalty += corner_extra_scale * (margin - dist_x) * (margin - dist_y)
        return penalty

    def get_evacuation_reward(self, scale=-0.1):
        """
        Reward for guide training: negative mean distance of evacuees to nearest exit.
        scale < 0 so that reducing distance gives positive reward. Returns 0 if no evacuees.
        """
        if not self.Exit:
            return 0.0
        total = 0.0
        n = 0
        for c in self.Cells:
            for p in c.Particles:
                if getattr(p, 'is_guide', False):
                    continue
                d = np.inf
                for e in self.Exit:
                    d = min(d, np.sqrt(np.sum((p.position - e) ** 2)))
                total += d
                n += 1
        if n == 0:
            return 0.0
        return scale * (total / n)

    def get_guide_reward_toward_crowd(self, reward_toward_crowd_scale=0.1, n_in_range_count_threshold=1, go_find_alone_bonus_scale=2.0):
        """
        Reward for moving toward evacuee centroid when 圈内人数 < threshold (go_find regime).
        When 圈内人数 == 0 (alone), multiply by go_find_alone_bonus_scale so go_find gets stronger signal.
        Call after step_guided().
        """
        guide_pos = None
        guide_velocity = None
        for c in self.Cells:
            for p in c.Particles:
                if getattr(p, 'is_guide', False):
                    guide_pos = p.position.copy()
                    guide_velocity = np.array(p.velocity[:2])
                    break
            if guide_pos is not None:
                break
        if guide_pos is None:
            return 0.0
        n_in_range_raw = self._count_evacuees_in_guide_range_raw(guide_pos)
        if n_in_range_raw >= n_in_range_count_threshold:
            return 0.0
        # Compute centroid of all evacuees (non-guide)
        sx, sy, n = 0.0, 0.0, 0
        for c in self.Cells:
            for p in c.Particles:
                if getattr(p, 'is_guide', False):
                    continue
                sx += p.position[0]
                sy += p.position[1]
                n += 1
        if n == 0:
            return 0.0
        cx, cy = sx / n, sy / n
        dx = cx - guide_pos[0]
        dy = cy - guide_pos[1]
        dist = np.sqrt(dx * dx + dy * dy) + 1e-10
        dir_to_crowd = np.array([dx / dist, dy / dist], dtype=np.float64)
        speed_xy = np.sqrt(np.sum(guide_velocity ** 2)) + 1e-10
        if speed_xy < 1e-8:
            return 0.0
        vel_dir = guide_velocity / speed_xy
        raw = reward_toward_crowd_scale * float(np.dot(vel_dir, dir_to_crowd))
        if n_in_range_raw == 0:
            raw *= go_find_alone_bonus_scale
        return raw

    def get_guide_dense_reward(self, n_in_range_count_threshold=1, reward_toward_exit_scale=0.1):
        """
        Merged dense reward for guide: 人数 × (朝向出口的A*进展).
        Call after step_guided(). No per-evacuee distance; evacuees follow guide, so guide's
        A* direction and velocity represent progress. When 圈内人数 < threshold return 0.
        """
        guide_pos = None
        guide_velocity = None
        for c in self.Cells:
            for p in c.Particles:
                if getattr(p, 'is_guide', False):
                    guide_pos = p.position.copy()
                    guide_velocity = np.array(p.velocity[:2])
                    break
            if guide_pos is not None:
                break
        if guide_pos is None or not self.Exit:
            return 0.0
        n_in_range = self._count_evacuees_in_guide_range_raw(guide_pos)
        if n_in_range < n_in_range_count_threshold:
            return 0.0
        speed_xy = np.sqrt(np.sum(guide_velocity ** 2)) + 1e-10
        if speed_xy < 1e-8:
            return 0.0
        vel_dir = guide_velocity / speed_xy
        _, astar_dx, astar_dy = self._astar_distance_and_direction_from_xy(guide_pos[0], guide_pos[1])
        dir_to_exit = np.array([astar_dx, astar_dy], dtype=np.float64)
        n_to = np.linalg.norm(dir_to_exit) + 1e-10
        if n_to < 1e-8:
            return 0.0
        dir_to_exit = dir_to_exit / n_to
        # Product: n_in_range * (velocity alignment with A* to exit)
        alignment = float(np.dot(vel_dir, dir_to_exit))
        return reward_toward_exit_scale * n_in_range * alignment

    def _is_guided(self, particle):
        for c in self.Cells:
            for p in c.Particles:
                if p.ID == particle.ID:
                    continue

                if not getattr(p, 'is_guide', False):
                    continue

                dist = np.sqrt(np.sum((p.position - particle.position) ** 2))
                if dist <= self.guide_radius:
                    return True

        return False

    def step_guided(self, guide_actions=None, max_guide_speed=2.0):
        """
        Step function for guided evacuation.

        - Guide agents (is_guide=True): only collision logic (walls, obstacles, friction).
          No movement policy; they are intended to be driven by NN later.
        - Evacuee particles: collision avoidance, then exit-directed or KNN+noise movement
          (rush toward exit when near exit or near a guide; else KNN + noise).
        """
        done = False
        self._n_escaped_this_step = 0

        if self.Number == 0:
            done = True
            return done

        self.Zero_acc()
        # Visibility update is expensive (BFS per cell); do every 5 steps
        if getattr(self, '_debug_step_count', 0) % 5 == 0:
            self.update_visibility_system()

        self.region_confine()
        self.loop_cells()
        self.loop_neighbors()

        debug_step = getattr(self, '_debug_step_count', 0)
        self._debug_step_count = debug_step + 1

        # Cache guide positions once (avoid O(N^2) _is_guided / _get_guiding_guide_position per step)
        _guide_positions = []
        for _c in self.Cells:
            for _p in _c.Particles:
                if getattr(_p, 'is_guide', False):
                    _guide_positions.append(_p.position.copy())

        guide_idx = 0
        for c in self.Cells:
            for p in c.Particles:
                if getattr(p, 'is_guide', False):
                    if guide_actions is not None and guide_idx < len(guide_actions) and len(guide_actions[guide_idx]) >= 2:
                        vx, vy = float(guide_actions[guide_idx][0]), float(guide_actions[guide_idx][1])
                        scale = np.sqrt(vx * vx + vy * vy) + 1e-10
                        if scale > 1:
                            vx, vy = vx / scale, vy / scale
                        target_vel = np.array([vx * max_guide_speed, vy * max_guide_speed, 0.0])
                        p.acc += (target_vel - np.array(p.velocity)) / relaxation_time
                    guide_idx += 1
                    continue

                # Evacuee particles: collision avoidance + exit / KNN + noise movement + guide memory
                exit_vector, exit_dist = self._nearest_exit_vector(p)
                is_guided_by_agent = False
                for gp in _guide_positions:
                    d = np.sqrt(np.sum((p.position[:2] - gp[:2]) ** 2))
                    if d <= self.guide_radius:
                        is_guided_by_agent = True
                        break

                # Update memory when inside guide radius
                current_step = getattr(self, '_debug_step_count', 0)
                n_interval = self.memory_astar_update_interval_n
                if is_guided_by_agent:
                    # When near guide: refresh A* direction every n steps if memory_strength >= thres_around
                    m = float(p.memory_strength)
                    last_up = getattr(p, 'last_astar_update_step', -999)
                    if m >= self.memory_astar_thres_around and (current_step - last_up) >= n_interval:
                        astar_dir = self._astar_direction_to_exit(p)
                        if astar_dir is not None:
                            mem_dir = np.array(astar_dir, dtype=np.float64)
                            mem_dir[2] = 0.0
                            norm_mem = np.linalg.norm(mem_dir[:2]) + 1e-10
                            mem_dir[:2] = mem_dir[:2] / norm_mem
                            p.exit_path_memory = mem_dir
                            p.last_astar_update_step = current_step
                    # Mark that this evacuee has been guided at least once
                    p.first_guided = True
                    # If first-guided bonus has not been consumed yet, flag this step
                    if getattr(p, "needs_first_memory_reward", True):
                        p.just_guided_this_step = True
                    # Memory quickly increases to 1 when near the guide
                    p.memory_strength = min(
                        1.0,
                        float(p.memory_strength) + self.memory_increase_rate * self.dt,
                    )
                else:
                    # Away from guide: memory slowly decays
                    if p.memory_strength > 0.0:
                        p.memory_strength = max(
                            0.0,
                            float(p.memory_strength) - self.memory_decay_rate * self.dt,
                        )
                    # When not near guide: refresh A* direction every n steps if memory_strength >= thres_out
                    m = float(p.memory_strength)
                    last_up = getattr(p, 'last_astar_update_step', -999)
                    if m >= self.memory_astar_thres_out and (current_step - last_up) >= n_interval:
                        astar_dir = self._astar_direction_to_exit(p)
                        if astar_dir is not None:
                            mem_dir = np.array(astar_dir, dtype=np.float64)
                            mem_dir[2] = 0.0
                            norm_mem = np.linalg.norm(mem_dir[:2]) + 1e-10
                            mem_dir[:2] = mem_dir[:2] / norm_mem
                            p.exit_path_memory = mem_dir
                            p.last_astar_update_step = current_step

                # Collision avoidance when far from exits
                collision_avoid_dir = None
                if exit_dist > self.door_visible_radius:
                    collision_avoid_dir = self._get_collision_avoidance_direction(p)

                best_action = None
                desired_speed = 0.0

                _scale = self.guide_speed_scale if getattr(p, 'is_guide', False) else self.speed_scale
                if collision_avoid_dir is not None:
                    # High-priority: avoid collisions, ignore memory mixing
                    move_vector = collision_avoid_dir
                    desired_speed = desire_velocity * _scale * 1.2
                    best_action = self._best_action_for_direction(move_vector)
                elif exit_dist <= self.door_visible_radius:
                    # Near exits: go directly toward exit
                    move_vector = exit_vector
                    desired_speed = desire_velocity * _scale
                    best_action = self._best_action_for_direction(move_vector)
                else:
                    # Far from exits: KNN + noise, mixed with remembered A* direction from guide
                    if self.use_knn:
                        search_cells = [c] + [self.Cells[n] for n in c.Neighbors]
                        move_vector_knn, speed_variance = self._knn_direction_and_variance(
                            p, self.knn_k, search_cells=search_cells
                        )
                        noise = np.random.normal(0.0, np.sqrt(speed_variance * 2.0))
                        desired_speed_knn = max(
                            0.1, (desire_velocity + noise) * _scale
                        )
                        # Base KNN velocity vector (2D)
                        v_knn_raw = desired_speed_knn * np.array(
                            [move_vector_knn[0], move_vector_knn[1]], dtype=np.float64
                        )
                    else:
                        # No KNN: random direction, treat as KNN-equivalent
                        action_idx = np.random.choice(len(self.action))
                        rand_dir = self.action[action_idx]
                        move_vector_knn = np.array(rand_dir, dtype=np.float64)
                        desired_speed_knn = desire_velocity * _scale
                        v_knn_raw = desired_speed_knn * np.array(
                            [move_vector_knn[0], move_vector_knn[1]], dtype=np.float64
                        )

                    m = float(p.memory_strength)
                    # Remembered A* direction component
                    v_astar = np.zeros(2, dtype=np.float64)
                    if m > 0.0:
                        mem_vec = np.array(p.exit_path_memory, dtype=np.float64)
                        mem_vec[2] = 0.0
                        norm_mem = np.linalg.norm(mem_vec[:2])
                        if norm_mem > 1e-8:
                            mem_dir_2d = mem_vec[:2] / norm_mem
                            base_speed = desire_velocity * _scale
                            v_astar = m * base_speed * mem_dir_2d

                    # Mix remembered path and KNN+noise: v_total = v_astar + (1-m) * v_knn_raw
                    v_total_2d = v_astar + (1.0 - m) * v_knn_raw
                    total_speed = np.linalg.norm(v_total_2d)
                    if total_speed < 1e-8:
                        # Fallback: use raw KNN velocity if mixing degenerates
                        v_total_2d = v_knn_raw
                        total_speed = np.linalg.norm(v_total_2d) + 1e-10

                    move_vector = np.array(
                        [v_total_2d[0] / total_speed, v_total_2d[1] / total_speed, 0.0],
                        dtype=np.float64,
                    )
                    desired_speed = total_speed
                    best_action = self._best_action_for_direction(move_vector)

                if best_action is None:
                    continue

                p.acc += 1 / relaxation_time * desired_speed * best_action

        # Clamp guide velocity BEFORE Integration to prevent "flashing" from large wall/obstacle forces
        # This ensures position updates use bounded velocity
        for c in self.Cells:
            for p in c.Particles:
                if getattr(p, 'is_guide', False):
                    speed = np.sqrt(p.velocity[0]**2 + p.velocity[1]**2) + 1e-10
                    if speed > max_guide_speed:
                        p.velocity[0] = p.velocity[0] * (max_guide_speed / speed)
                        p.velocity[1] = p.velocity[1] * (max_guide_speed / speed)
        
        self.Integration(1)
        self.Integration(0)
        self.move_particles()
        
        # Clamp guide velocity again AFTER Integration to ensure it stays within bounds
        # (in case wall/obstacle forces added during Integration caused speed to exceed limit)
        for c in self.Cells:
            for p in c.Particles:
                if getattr(p, 'is_guide', False):
                    speed = np.sqrt(p.velocity[0]**2 + p.velocity[1]**2) + 1e-10
                    if speed > max_guide_speed:
                        p.velocity[0] = p.velocity[0] * (max_guide_speed / speed)
                        p.velocity[1] = p.velocity[1] * (max_guide_speed / speed)
        
        self._correct_obstacle_penetration()
        self._remove_particles_at_exits()

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

