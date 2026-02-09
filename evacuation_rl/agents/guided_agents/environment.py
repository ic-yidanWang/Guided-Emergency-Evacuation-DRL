"""
Guided Agent Environment

This module extends the base cellspace environment to support guided evacuation scenarios:
1. Agents near exits have knowledge of the optimal exit route
2. Other agents follow crowd behavior patterns
3. Social following: agents tend to move in the direction where many others are moving
4. Guide agents can provide directional cues to help evacuation
"""

import numpy as np
from evacuation_rl.environments.cellspace import Cell_Space, Particle


class GuidedParticle(Particle):
    """
    Extended particle for guided evacuation scenarios
    
    Attributes:
        knows_exit: Whether this agent knows the location of the nearest exit
        follow_threshold: Velocity threshold for following crowd behavior
        social_weight: Weight given to social following behavior
    """
    
    def __init__(self, ID, x, y, z, vx, vy, vz, mass=80.0, type=1, 
                 knows_exit=False, follow_threshold=1.0, social_weight=0.5):
        super().__init__(ID, x, y, z, vx, vy, vz, mass, type)
        self.knows_exit = knows_exit
        self.follow_threshold = follow_threshold
        self.social_weight = social_weight


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
                 exit_knowledge_radius=2.0):
        super().__init__(xmin, xmax, ymin, ymax, zmin, zmax, rcut, dt, Number)
        self.exit_knowledge_radius = exit_knowledge_radius
    
    def update_exit_knowledge(self):
        """Update which agents know about exits based on proximity"""
        for c in self.Cells:
            for p in c.Particles:
                # Check if agent is close enough to any exit to know about it
                for e in self.Exit:
                    dis = np.sqrt(np.sum((p.position - e) ** 2))
                    if dis < self.exit_knowledge_radius:
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
    
    # TODO: Implement step_guided method for guided evacuation
    # TODO: Implement guide_agent behavior
    # TODO: Add visualization for knowledge distribution


# Future work:
# - Implement guide agent DQN network
# - Add training loop for guide agents
# - Implement multi-agent reinforcement learning for guided evacuation
# - Add metrics for evacuation efficiency with/without guides
