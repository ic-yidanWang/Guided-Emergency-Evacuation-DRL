"""
Guided Agent Environment

This module extends the base cellspace environment to support guided evacuation scenarios:
1. Agents near exits have knowledge of the optimal exit route
2. Other agents follow crowd behavior patterns
3. Social following: agents tend to move in the direction where many others are moving
4. Guide agents can provide directional cues to help evacuation
"""

import numpy as np
from evacuation_rl.environments import cellspace
from evacuation_rl.environments.cellspace import (
    Cell_Space,
    Particle,
    agent_size,
    desire_velocity,
    relaxation_time,
)


class GuidedParticle(Particle):
    """
    Extended particle for guided evacuation scenarios
    
    Attributes:
        knows_exit: Whether this agent knows the location of the nearest exit
        follow_threshold: Velocity threshold for following crowd behavior
        social_weight: Weight given to social following behavior
        is_guide: Whether this agent acts as a guide
    """
    
    def __init__(self, ID, x, y, z, vx, vy, vz, mass=80.0, type=1,
                 knows_exit=False, follow_threshold=1.0, social_weight=0.5,
                 is_guide=False):
        super().__init__(ID, x, y, z, vx, vy, vz, mass, type)
        self.knows_exit = knows_exit
        self.follow_threshold = follow_threshold
        self.social_weight = social_weight
        self.is_guide = is_guide


class GuidedCellSpace(Cell_Space):
    def set_main_agent_action(self, action):
        """
        Set the action for the main agent (to be used by DQN only).
        This should be called before step_guided().
        """
        self._main_agent_action = action

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
                 n_move_guide=1, guide_radius=1, use_knn=True, speed_scale=1.0,
                 n_static_guide=4, obstacle_configs=None, knn_max_distance=3.0,
                 knn_filter_obstacles=True):
        self.n_move_guide = max(0, int(n_move_guide))
        # 速度上限，默认2.0，可通过config传入
        self.max_velocity = None
        # agents list for RL/main agents, compatible with Cell_Space
        self.agents = []
        self.guide_radius = guide_radius
        self.n_static_guide = n_static_guide
        # Store obstacle configs BEFORE calling parent __init__ (which calls initialize_particles)
        self.obstacle_configs = obstacle_configs if obstacle_configs is not None else []
        super().__init__(xmin, xmax, ymin, ymax, zmin, zmax, rcut, dt, Number)
        # door_visible_radius: distance at which agents start rushing toward exit directly
        self.door_visible_radius = door_visible_radius
        self.knn_k = max(1, int(knn_k))
        self.use_knn = use_knn
        self.speed_scale = max(0.1, float(speed_scale))
        self.knn_max_distance = knn_max_distance
        self.knn_filter_obstacles = knn_filter_obstacles
        
        # Initialize guide points at random positions
        self._initialize_guide_points()
    
    def _check_obstacle_collision(self, pos, agent_size):
        """Check if position collides with any obstacle (using true geometric shapes)"""
        # All coordinates are now in absolute format
        
        # Check against original obstacle configurations if available
        if hasattr(self, 'obstacle_configs') and self.obstacle_configs:
            for obs_cfg in self.obstacle_configs:
                obs_type = obs_cfg.get('type', 'circle')
                
                if obs_type == 'circle':
                    # Check if inside circle - coordinates are in absolute units
                    center = np.array([obs_cfg['x'], obs_cfg['y'], obs_cfg.get('z', 0.5)])
                    radius = obs_cfg.get('size', 0.5)  # Radius in absolute units
                    
                    dis = np.sqrt(np.sum((pos - center) ** 2))
                    if dis < radius + agent_size:
                        print(f"[DEBUG] Particle at {pos} is inside circle obstacle at {center} (r={radius}), agent_size={agent_size}")
                        return True
                
                elif obs_type == 'rectangle':
                    # Check if inside rectangle (absolute coordinates)
                    center_x = obs_cfg['x']
                    center_y = obs_cfg['y']
                    width = obs_cfg.get('width', 0.4)
                    height = obs_cfg.get('height', 0.3)
                    
                    # Rectangle boundaries (absolute coordinates)
                    left = center_x - width/2
                    right = center_x + width/2
                    bottom = center_y - height/2
                    top = center_y + height/2
                    
                    # Check if point is inside rectangle
                    if (left - agent_size <= pos[0] <= right + agent_size and 
                        bottom - agent_size <= pos[1] <= top + agent_size):
                        print(f"[DEBUG] Particle at {pos} is inside rectangle obstacle center=({center_x},{center_y}), w={width}, h={height}, agent_size={agent_size}")
                        return True
        
        # Fallback: check discrete obstacle points
        for idx, ob in enumerate(self.Ob):
            if isinstance(ob, list):
                # Multiple points (e.g., rectangle grid)
                for p in ob:
                    dis = np.sqrt(np.sum((pos - p) ** 2))
                    if dis < (agent_size + self.Ob_size[idx]) / 2:
                        print(f"[DEBUG] Particle at {pos} is inside obstacle point {p}, agent_size={agent_size}, Ob_size={self.Ob_size[idx]}")
                        return True
            else:
                # Single point (e.g., circle center)
                dis = np.sqrt(np.sum((pos - ob) ** 2))
                if dis < (agent_size + self.Ob_size[idx]) / 2:
                    return True
        
        return False
    
    def _compute_obstacle_force(self, particle_pos, agent_size):
        """Compute repulsive force from obstacles using geometric shapes"""
        from evacuation_rl.environments.cellspace import f_collision_lim
        
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
                    dis_to_center = np.sqrt(np.sum(dr_to_center ** 2)) + 1e-10
                    
                    # Distance from particle to circle BOUNDARY (negative if inside)
                    dis_to_boundary = dis_to_center - radius
                    
                    # Only apply force if agent is close to or inside the circle
                    # Force should push agent away from circle center
                    if dis_to_boundary < agent_size:
                        # dis_eq is the equilibrium distance (agent should stay this far from boundary)
                        dis_eq = agent_size
                        penetration = dis_eq - dis_to_boundary
                        
                        # Apply repulsive force (direction: away from center)
                        f = f_collision_lim * np.exp(penetration / 0.08)
                        f_vec = f * dr_to_center / dis_to_center
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
                    
                    # Find closest point on rectangle SURFACE to the particle
                    closest_x = np.clip(particle_pos[0], left, right)
                    closest_y = np.clip(particle_pos[1], bottom, top)
                    closest = np.array([closest_x, closest_y, particle_pos[2]])
                    
                    # Distance from particle to closest point on rectangle
                    dr = particle_pos - closest
                    dis = np.sqrt(np.sum(dr ** 2)) + 1e-10
                    
                    # Only apply force if very close to rectangle
                    dis_eq = agent_size * 1.5  # Slightly larger buffer for rectangles
                    
                    if dis < dis_eq:
                        # Apply repulsive force
                        penetration = dis_eq - dis
                        f = f_collision_lim * np.exp(penetration / 0.08)
                        if dis > 1e-6:
                            f_vec = f * dr / dis
                        else:
                            # If agent is exactly on rectangle boundary, push outward
                            # Determine which edge is closest
                            if abs(particle_pos[0] - left) < 0.01:
                                f_vec = f * np.array([-1.0, 0.0, 0.0])
                            elif abs(particle_pos[0] - right) < 0.01:
                                f_vec = f * np.array([1.0, 0.0, 0.0])
                            elif abs(particle_pos[1] - bottom) < 0.01:
                                f_vec = f * np.array([0.0, -1.0, 0.0])
                            elif abs(particle_pos[1] - top) < 0.01:
                                f_vec = f * np.array([0.0, 1.0, 0.0])
                            else:
                                f_vec = f * np.array([1.0, 0.0, 0.0])
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
                            f = f_collision_lim * np.exp((dis_eq - dis) / 0.08)
                            f_vec = f * dr / dis
                            total_force += f_vec
                else:
                    dr = particle_pos - ob
                    dis = np.sqrt(np.sum(dr ** 2)) + 1e-10
                    dis_eq = (agent_size + self.Ob_size[idx]) / 2
                    
                    if dis < dis_eq:
                        f = f_collision_lim * np.exp((dis_eq - dis) / 0.08)
                        f_vec = f * dr / dis
                        total_force += f_vec
        
        return total_force
    
    def region_confine(self):
        """Apply region confining forces: walls, obstacles and friction (with geometric obstacles)"""
        from evacuation_rl.environments.cellspace import (
            f_wall_lim, agent_size, relaxation_time
        )
        
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
                
                # Obstacle forces using geometric shapes
                obstacle_force = self._compute_obstacle_force(p.position, agent_size)
                p.acc += obstacle_force / p.mass
                
                # Friction force
                f = -p.mass / relaxation_time * p.velocity
                p.acc += f / p.mass
    
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
    
    def _initialize_guide_points(self):
        """Initialize guide points at random positions in normalized coordinates [0, 1]"""
        # Generate random positions for guide points
        # Using normalized coordinates [0, 1] for consistency with Exit and Ob
        padding = 0.1  # Keep guides away from edges
        
        for i in range(self.n_static_guide):
            # Generate random position in normalized space [0, 1]
            guide_pos = np.array([
                padding + np.random.rand() * (1 - 2 * padding),  # x in [0.1, 0.9]
                padding + np.random.rand() * (1 - 2 * padding),  # y in [0.1, 0.9]
                0.5  # z in middle
            ])
            
            # Check if not too close to exits (to avoid confusion)
            too_close_to_exit = False
            for e in self.Exit:
                # Convert exit to normalized coordinates for comparison
                e_norm = (e - self.L[:, 0]) / (self.L[:, 1] - self.L[:, 0])
                dis = np.sqrt(np.sum((guide_pos - e_norm) ** 2))
                if dis < 0.15:  # Minimum distance from exits in normalized space
                    too_close_to_exit = True
                    break
            
            if not too_close_to_exit:
                self.Guide.append(self.L[:, 0] + guide_pos * (self.L[:, 1] - self.L[:, 0]))
        
        print(f"Initialized {len(self.Guide)} guide points at random positions.")

    def initialize_particles(self, file=None, n_agents=1):
        """Initialize guided particles with main agent(s) and guided agents, compatible with agents list."""
        print("Initializing particles.")
        self.main_agent = None
        self.guided_agents = []
        self.agents = []
        if file is None:
            padding = 0.05  # 5% of domain
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
                x_min = self.L[0, 0] + padding + i * cell_w
                x_max = x_min + cell_w
                y_min = self.L[1, 0] + padding + j * cell_h
                y_max = y_min + cell_h
                z = self.L[2, 0] + 0.5 * (self.L[2, 1] - self.L[2, 0])
                eps = 1e-6
                x = np.random.uniform(x_min + eps, x_max - eps) if x_max > x_min + 2*eps else 0.5 * (x_min + x_max)
                y = np.random.uniform(y_min + eps, y_max - eps) if y_max > y_min + 2*eps else 0.5 * (y_min + y_max)
                initial_pos = np.array([x, y, z])
                pos = self._find_valid_position(initial_pos, positions, agent_size)
                if not self._check_obstacle_collision(pos, agent_size):
                    positions.append(pos)
            if len(positions) < self.Number:
                self.Number = len(positions)
                self.Total = len(positions)
            for i in range(self.Number):
                pos = positions[i]
                near_exit_distance = 1.5
                v = np.array([0., 0., 0.])
                nearest_exit = None
                nearest_exit_dist = np.inf
                for e in self.Exit:
                    dist = np.linalg.norm(pos - e)
                    if dist < nearest_exit_dist:
                        nearest_exit_dist = dist
                        nearest_exit = e
                if nearest_exit is not None and nearest_exit_dist < near_exit_distance:
                    exit_direction = nearest_exit - pos
                    exit_direction[2] = 0
                    exit_dir_norm = np.linalg.norm(exit_direction)
                    if exit_dir_norm > 1e-6:
                        v = (exit_direction / exit_dir_norm) * 0.2
                    else:
                        v = np.random.randn(3) * 0.01
                else:
                    v = np.random.randn(3) * 0.01
                v[2] = 0.
                v = v.tolist()
                is_guide = i < min(self.n_move_guide, self.Number)
                particle = GuidedParticle(i, *pos.tolist(), *v, is_guide=is_guide)
                if i < n_agents:
                    if self.main_agent is None:
                        self.main_agent = particle
                    self.agents.append(particle)
                else:
                    self.guided_agents.append(particle)
                self.insert_particle(particle)
        else:
            super().initialize_particles(file=file, n_agents=n_agents)
        print(f"The number of agents is {self.Number}.")
    
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

    def _knn_direction_and_variance(self, particle, k):
        """
        Get direction informed by KNN neighbors with enhanced randomness.
        
        Behavior for agents who DON'T see exits yet:
        1. Follow neighbors primarily (60%)
        2. Random walk component (40%)
        3. Large speed variance noise for realistic uncertainty
        
        KNN filtering:
        - Only considers neighbors within knn_max_distance
        - If knn_filter_obstacles is True, excludes neighbors blocked by obstacles
        """
        neighbors = []

        for c in self.Cells:
            for p in c.Particles:
                if p.ID == particle.ID:
                    continue

                dist = np.sqrt(np.sum((p.position - particle.position) ** 2))
                
                # Filter 1: Maximum distance threshold
                if dist > self.knn_max_distance:
                    continue
                
                # Filter 2: Line of sight blocked by obstacles
                if self.knn_filter_obstacles and self._is_line_of_sight_blocked(particle.position, p.position):
                    continue
                
                neighbors.append((dist, p))

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

    def step_conformity(self):
        """
        Step function for conformity agents.

        Behavior:
        - Agents within door_threshold move toward nearest exit
        - Others follow KNN average direction
        - Speed noise is added based on neighbors' speed variance
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
                exit_vector, exit_dist = self._nearest_exit_vector(p)

                if exit_dist <= self.door_threshold:
                    move_vector = exit_vector
                    desired_speed = desire_velocity
                else:
                    if self.use_knn:
                        move_vector, speed_variance = self._knn_direction_and_variance(p, self.knn_k)
                        noise = np.random.normal(0.0, np.sqrt(speed_variance)) if speed_variance > 0.0 else 0.0
                        desired_speed = max(0.1, (desire_velocity + noise) * self.speed_scale)
                        best_action = self._best_action_for_direction(move_vector)
                    else:
                        action_idx = np.random.choice(len(self.action))
                        best_action = self.action[action_idx]
                        desired_speed = desire_velocity * self.speed_scale

                if best_action is None:
                    continue

                p.acc += 1 / relaxation_time * desired_speed * best_action

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
                    if dis < cellspace.dis_lim:
                        c.Particles.pop(i)
                        in_exit = True
                        self.Number -= 1
                        break

                if not in_exit:
                    i += 1

        if self.Number == 0:
            done = True

        return done

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

    def step_guided(self):
        """
        Step function for guided evacuation with two-stage behavior.

        Behavior:
        - If near guides (within guide_radius): rush toward nearest exit
        - Else if within door_visible_radius: rush toward nearest exit  
        - Else: follow KNN average direction (or random if KNN disabled)
          * Except if near guides, then still influenced by them
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

        debug_step = getattr(self, '_debug_step_count', 0)
        self._debug_step_count = debug_step + 1

        
        main_action = getattr(self, '_main_agent_action', None) # DQN-only logic for main agent
        # travel thorugh all cells and particles, apply DQN action to main agent, and normal logic to others
        for c in self.Cells:
            for p in c.Particles:
                if self.main_agent is not None and p.ID == self.main_agent.ID:
                    # Only DQN controls the main agent
                    if main_action is not None:
                        best_action = self.action[main_action] if hasattr(self, 'action') else None
                        desired_speed = desire_velocity * self.speed_scale
                        if best_action is not None:
                            p.acc += 1 / relaxation_time * desired_speed * best_action
                    continue
                # Normal logic for other particles (unchanged)
                exit_vector, exit_dist = self._nearest_exit_vector(p)
                is_guided_by_agent = self._is_guided(p)
                collision_avoid_dir = None
                if exit_dist > self.door_visible_radius:
                    collision_avoid_dir = self._get_collision_avoidance_direction(p)
                if collision_avoid_dir is not None:
                    move_vector = collision_avoid_dir
                    desired_speed = desire_velocity * self.speed_scale * 1.2
                    best_action = self._best_action_for_direction(move_vector)
                elif getattr(p, 'is_guide', False) or is_guided_by_agent:
                    move_vector = exit_vector
                    desired_speed = desire_velocity * self.speed_scale
                    best_action = self._best_action_for_direction(move_vector)
                elif exit_dist <= self.door_visible_radius:
                    move_vector = exit_vector
                    desired_speed = desire_velocity * self.speed_scale
                    best_action = self._best_action_for_direction(move_vector)
                else:
                    if self.use_knn:
                        move_vector, speed_variance = self._knn_direction_and_variance(p, self.knn_k)
                        noise = np.random.normal(0.0, np.sqrt(speed_variance * 2.0))
                        desired_speed = max(0.1, (desire_velocity + noise) * self.speed_scale)
                        best_action = self._best_action_for_direction(move_vector)
                    else:
                        action_idx = np.random.choice(len(self.action))
                        best_action = self.action[action_idx]
                        desired_speed = desire_velocity * self.speed_scale
                if best_action is None:
                    continue
                p.acc += 1 / relaxation_time * desired_speed * best_action

        self.Integration(1, max_velocity=self.max_velocity)
        self.Integration(0, max_velocity=self.max_velocity)
        self.move_particles()

        for c in self.Cells:
            i = 0
            while i < len(c.Particles):
                in_exit = False
                for e in self.Exit:
                    dis = c.Particles[i].position - e
                    dis = np.sqrt(np.sum(dis ** 2))
                    if dis < cellspace.dis_lim:
                        c.Particles.pop(i)
                        in_exit = True
                        self.Number -= 1
                        break
                if not in_exit:
                    i += 1

        if self.Number == 0:
            done = True

        return done
    
    # TODO: Implement step_guided method for guided evacuation
    # TODO: Implement guide_agent behavior
    # TODO: Add visualization for knowledge distribution


# Future work:
# - Implement guide agent DQN network
# - Add training loop for guide agents
# - Implement multi-agent reinforcement learning for guided evacuation
# - Add metrics for evacuation efficiency with/without guides
