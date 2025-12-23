import numpy as np
from numba import jit

@jit(nopython=True, fastmath=True)
def ray_segment_intersect(ray_origin_x, ray_origin_y, ray_dx, ray_dy, seg_x1, seg_y1, seg_x2, seg_y2):
    # Ray: P + t * D
    # Segment: A + u * (B - A)
    # t > 0, 0 <= u <= 1
    
    seg_dx = seg_x2 - seg_x1
    seg_dy = seg_y2 - seg_y1
    
    # Cross product 2D (determinant)
    det = ray_dx * seg_dy - ray_dy * seg_dx
    
    if abs(det) < 1e-6:
        return 2.0 # Parallel
        
    t_num = (seg_x1 - ray_origin_x) * seg_dy - (seg_y1 - ray_origin_y) * seg_dx
    u_num = (seg_x1 - ray_origin_x) * ray_dy - (seg_y1 - ray_origin_y) * ray_dx
    
    t = t_num / det
    u = u_num / det
    
    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t
    return 2.0 # No hit

@jit(nopython=True, fastmath=True)
def ray_circle_intersect(ray_origin_x, ray_origin_y, ray_dx, ray_dy, circ_x, circ_y, radius):
    # Ray: P + t * D. Assume D is normalized.
    # |P + tD - C|^2 = r^2
    # Let L = C - P
    # t^2 - 2(L.D)t + L.L - r^2 = 0
    
    Lx = circ_x - ray_origin_x
    Ly = circ_y - ray_origin_y
    
    L_dot_D = Lx * ray_dx + Ly * ray_dy
    
    # If ray is pointing away from circle (and outside), reject early? 
    # But carefully - we might be inside? No, we start outside.
    
    c = (Lx * Lx + Ly * Ly) - radius * radius
    
    # Discriminant
    # D^2 = (2(L.D))^2 - 4(1)(c) = 4(L.D)^2 - 4c
    # We want sqrt(D^2) -> 2 sqrt((L.D)^2 - c)
    # roots = (2(L.D) +/- 2 sqrt(...)) / 2 = L.L +/- sqrt(...)
    
    disc = L_dot_D * L_dot_D - c
    
    if disc < 0:
        return 2.0 # No intersection
        
    sqrt_disc = np.sqrt(disc)
    t1 = L_dot_D - sqrt_disc
    t2 = L_dot_D + sqrt_disc
    
    # We want smallest positive t
    if t1 >= 0:
        return t1
    if t2 >= 0:
        return t2
        
    return 2.0


@jit(nopython=True, parallel=False, fastmath=True)
def compute_sensors_batch(agent_states, wall_segments, obstacle_circles, n_rays, range_max):
    """
    agent_states: (N, 4) -> [x, y, angle, radius]
    wall_segments: (W, 4) -> [x1, y1, x2, y2]
    obstacle_circles: (M, 3) -> [x, y, radius] (Usually same as agents)
    """
    n_agents = agent_states.shape[0]
    n_walls = wall_segments.shape[0]
    n_obstacles = obstacle_circles.shape[0]
    
    # Output: (N, n_rays)
    sensors = np.ones((n_agents, n_rays), dtype=np.float32) # Default 1.0 (normalized)
    
    # Angles linspace
    # We precalculate relative headings? No, each agent has different angle.
    
    two_pi = 2 * np.pi
    angle_step = two_pi / n_rays
    
    for i in range(n_agents):
        ax = agent_states[i, 0]
        ay = agent_states[i, 1]
        a_angle = agent_states[i, 2]
        a_radius = agent_states[i, 3] # Not used for ray origin offset if we assume point start? 
        # But we should start ray from edge of agent to avoid self-hit.
        
        offset_dist = a_radius + 1.0
        
        for r in range(n_rays):
            # Global ray angle
            ray_angle = a_angle + r * angle_step
            
            # Direction
            rdx = np.cos(ray_angle)
            rdy = np.sin(ray_angle)
            
            # Start pos
            rox = ax + rdx * offset_dist
            roy = ay + rdy * offset_dist
            
            # Max travel
            min_dist = range_max
            
            # Check Walls
            for w in range(n_walls):
                # Normalized distance (t) is in units of Ray Length? 
                # ray_segment_intersect assumes D is a vector. If D is normalized, t is distance.
                dist = ray_segment_intersect(rox, roy, rdx, rdy, 
                                          wall_segments[w, 0], wall_segments[w, 1], 
                                          wall_segments[w, 2], wall_segments[w, 3])
                if dist < min_dist:
                    min_dist = dist
                    
            # Check Obstacles (Agents)
            for o in range(n_obstacles):
                # Dont check self
                # We can check index or distance. 
                # Using distance check: if origin is inside obstacle?
                # Using index is safer if passed.
                # Let's assume passed index matching? 
                # Simplest: if (ax-ox)^2 + (ay-oy)^2 < eps, skip.
                
                ox = obstacle_circles[o, 0]
                oy = obstacle_circles[o, 1]
                orad = obstacle_circles[o, 2]
                
                dx = ax - ox
                dy = ay - oy
                if (dx*dx + dy*dy) < 1.0:
                    continue # It's me
                
                dist = ray_circle_intersect(rox, roy, rdx, rdy, ox, oy, orad)
                if dist < min_dist:
                    min_dist = dist
            
            # Normalize to 0-1
            sensors[i, r] = min_dist / range_max
            
    return sensors
