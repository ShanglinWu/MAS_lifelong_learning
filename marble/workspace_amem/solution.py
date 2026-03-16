# solution.py
import random
import time
import math
import heapq
import json
import threading
from enum import Enum
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

# ============================
# CONFIGURATION AND CONSTANTS
# ============================

# Game world dimensions
WORLD_WIDTH = 1000
WORLD_HEIGHT = 800

# Team configurations
TEAM_COUNT = 2
TEAM_NAMES = ["Red Team", "Blue Team"]
TEAM_COLORS = [(255, 0, 0), (0, 0, 255)]

# Robot configuration
MAX_ROBOTS_PER_TEAM = 5
ROBOT_MAX_HEALTH = 100
ROBOT_MAX_ENERGY = 100
ROBOT_SPEED = 2.0
ROBOT_ATTACK_COOLDOWN = 1.0  # seconds

# Weapon configurations
WEAPON_TYPES = {
    "pulse_rifle": {"damage": 15, "range": 300, "cooldown": 0.5, "type": "ranged", "ammo": 30},
    "plasma_cannon": {"damage": 30, "range": 400, "cooldown": 2.0, "type": "ranged", "ammo": 10},
    "melee_blade": {"damage": 25, "range": 50, "cooldown": 1.5, "type": "melee", "ammo": 0},
    "rocket_launcher": {"damage": 50, "range": 500, "cooldown": 4.0, "type": "ranged", "ammo": 3}
}

# Power-up configurations
POWER_UP_TYPES = {
    "health_pack": {"effect": "heal", "amount": 25, "duration": 0, "color": (0, 255, 0)},
    "energy_boost": {"effect": "energy", "amount": 30, "duration": 0, "color": (0, 100, 255)},
    "speed_boost": {"effect": "speed", "amount": 1.5, "duration": 8.0, "color": (255, 255, 0)},
    "damage_boost": {"effect": "damage", "amount": 1.5, "duration": 10.0, "color": (255, 0, 255)},
    "shield": {"effect": "shield", "amount": 50, "duration": 15.0, "color": (100, 100, 255)}
}

# Environmental hazards
HAZARD_TYPES = {
    "laser_grid": {"damage_per_second": 5, "radius": 80, "color": (255, 100, 0)},
    "electric_puddle": {"damage_per_second": 3, "radius": 60, "color": (100, 100, 255)},
    "toxic_gas": {"damage_per_second": 2, "radius": 100, "color": (150, 150, 0)}
}

# Objective types
class ObjectiveType(Enum):
    FLAG_CAPTURE = "flag_capture"
    BASE_DEFENSE = "base_defense"
    PAYLOAD_ESCORT = "payload_escort"
    CONTROL_POINT = "control_point"

# AI Behavior States
class AIState(Enum):
    IDLE = "idle"
    PATROL = "patrol"
    ATTACK = "attack"
    DEFEND = "defend"
    RETREAT = "retreat"
    CAPTURE = "capture"
    ESCORT = "escort"
    RELOAD = "reload"

# ============================
# DATA CLASSES
# ============================

@dataclass
class Vector2D:
    """Represents a 2D position or velocity vector."""
    x: float = 0.0
    y: float = 0.0
    
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar):
        return Vector2D(self.x / scalar, self.y / scalar)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)
    
    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)
    
    def distance_to(self, other):
        return (self - other).magnitude()
    
    def to_tuple(self):
        return (self.x, self.y)

@dataclass
class Robot:
    """Represents an AI-controlled robot in the arena."""
    id: int
    team: int
    position: Vector2D
    velocity: Vector2D = field(default_factory=lambda: Vector2D())
    health: float = ROBOT_MAX_HEALTH
    energy: float = ROBOT_MAX_ENERGY
    weapon: str = "pulse_rifle"
    weapon_ammo: int = WEAPON_TYPES["pulse_rifle"]["ammo"]
    weapon_cooldown: float = 0.0
    state: AIState = AIState.IDLE
    role: str = "scout"  # scout, attacker, defender, support
    target: Optional[Vector2D] = None
    last_seen_enemy: Optional[Vector2D] = None
    vision_range: float = 200.0
    last_action_time: float = 0.0
    power_ups: Dict[str, float] = field(default_factory=dict)  # power_up_name -> expiration_time
    armor: float = 0.0
    experience: float = 0.0  # for learning system
    kill_count: int = 0
    assist_count: int = 0
    objectives_completed: int = 0
    
    def update(self, delta_time: float, game_state: 'GameState') -> None:
        """Update robot state based on current game environment."""
        # Update weapon cooldown
        if self.weapon_cooldown > 0:
            self.weapon_cooldown -= delta_time
            self.weapon_cooldown = max(0, self.weapon_cooldown)
        
        # Update power-up durations
        current_time = game_state.current_time
        expired_powerups = []
        for powerup, expire_time in self.power_ups.items():
            if current_time >= expire_time:
                expired_powerups.append(powerup)
        
        for powerup in expired_powerups:
            del self.power_ups[powerup]
            # Reset any boosted stats
            if powerup == "speed_boost":
                self.vision_range = max(100, self.vision_range / 1.5)
            elif powerup == "damage_boost":
                # Weapon damage is handled in combat logic
                pass
            elif powerup == "shield":
                self.armor = 0
        
        # Apply speed boost if active
        if "speed_boost" in self.power_ups:
            speed_multiplier = WEAPON_TYPES[self.weapon].get("speed_multiplier", 1.0)
            base_speed = ROBOT_SPEED * 1.5 * speed_multiplier
        else:
            base_speed = ROBOT_SPEED
        
        # Update position based on velocity
        self.position += self.velocity * delta_time * base_speed
        
        # Keep robot within bounds
        self.position.x = max(0, min(WORLD_WIDTH, self.position.x))
        self.position.y = max(0, min(WORLD_HEIGHT, self.position.y))
        
        # Reset velocity if not moving
        if self.state == AIState.IDLE or self.state == AIState.RELOAD:
            self.velocity = Vector2D(0, 0)
        
        # Update experience based on performance
        self.experience += delta_time * 0.1  # Passive experience gain
        
        # Check for environmental hazards
        self.check_hazards(game_state)
    
    def check_hazards(self, game_state: 'GameState') -> None:
        """Check if robot is in any environmental hazard and take damage."""
        for hazard in game_state.hazards:
            distance = self.position.distance_to(hazard.position)
            if distance < hazard.radius:
                # Apply damage based on hazard type
                damage_per_second = hazard.damage_per_second
                damage = damage_per_second * 0.1  # Assuming 10fps update rate
                self.take_damage(damage, "environmental")
    
    def take_damage(self, damage: float, source: str = "enemy") -> None:
        """Apply damage to robot with armor mitigation."""
        if self.armor > 0:
            armor_absorption = min(self.armor, damage * 0.5)  # Armor absorbs 50% of damage up to its value
            damage -= armor_absorption
            self.armor -= armor_absorption
            self.armor = max(0, self.armor)
        
        self.health -= damage
        self.health = max(0, self.health)
        
        # If robot is defeated, trigger respawn or team update
        if self.health <= 0:
            self.health = 0
            # Robot is defeated, respawn after delay
            game_state.defeated_robots.append((self.id, time.time()))
    
    def heal(self, amount: float) -> None:
        """Heal robot by specified amount."""
        self.health = min(ROBOT_MAX_HEALTH, self.health + amount)
    
    def restore_energy(self, amount: float) -> None:
        """Restore robot energy."""
        self.energy = min(ROBOT_MAX_ENERGY, self.energy + amount)
    
    def use_weapon(self, target: Vector2D, game_state: 'GameState') -> bool:
        """Attempt to use current weapon on target. Returns True if successful."""
        if self.weapon_cooldown > 0:
            return False
        
        weapon_info = WEAPON_TYPES[self.weapon]
        
        # Check if target is in range
        distance = self.position.distance_to(target)
        if distance > weapon_info["range"]:
            return False
        
        # Check if enough energy for weapon
        energy_cost = 5
        if self.energy < energy_cost:
            return False
        
        # Check ammo for ranged weapons
        if weapon_info["ammo"] > 0 and self.weapon_ammo <= 0:
            return False
        
        # Apply weapon cooldown
        self.weapon_cooldown = weapon_info["cooldown"]
        self.energy -= energy_cost
        
        # Use ammo if applicable
        if weapon_info["ammo"] > 0:
            self.weapon_ammo -= 1
        
        # Create projectile or melee effect
        if weapon_info["type"] == "ranged":
            game_state.add_projectile(self.id, self.position, target, weapon_info["damage"], self.team)
        else:  # melee
            # Check for enemies in melee range
            for robot in game_state.robots:
                if robot.team != self.team and self.position.distance_to(robot.position) <= weapon_info["range"]:
                    robot.take_damage(weapon_info["damage"], f"robot_{self.id}")
                    # Add assist for teammates nearby
                    for teammate in game_state.robots:
                        if teammate.team == self.team and teammate.id != self.id:
                            if self.position.distance_to(teammate.position) < 150:
                                teammate.assist_count += 1
                    return True
        
        return True
    
    def pick_up_power_up(self, power_up: 'PowerUp') -> bool:
        """Pick up a power-up if within range."""
        if self.position.distance_to(power_up.position) < 40:
            # Apply power-up effect
            effect = POWER_UP_TYPES[power_up.type]
            if effect["effect"] == "heal":
                self.heal(effect["amount"])
            elif effect["effect"] == "energy":
                self.restore_energy(effect["amount"])
            elif effect["effect"] == "speed":
                self.power_ups["speed_boost"] = time.time() + effect["duration"]
                self.vision_range *= 1.5
            elif effect["effect"] == "damage":
                self.power_ups["damage_boost"] = time.time() + effect["duration"]
            elif effect["effect"] == "shield":
                self.armor = effect["amount"]
                self.power_ups["shield"] = time.time() + effect["duration"]
            
            return True
        return False
    
    def get_ai_decision(self, game_state: 'GameState') -> None:
        """AI decision-making core: determines robot's next action based on environment."""
        # Determine current threat level and objectives
        nearest_enemy = self.find_nearest_enemy(game_state)
        nearest_objective = self.find_nearest_objective(game_state)
        nearest_allied_robot = self.find_nearest_teammate(game_state)
        
        # Check if we're out of ammo and need to reload
        if (WEAPON_TYPES[self.weapon]["ammo"] > 0 and 
            self.weapon_ammo <= 0 and 
            self.state != AIState.RELOAD):
            self.state = AIState.RELOAD
            self.target = None
            return
        
        # Check if we need to retreat due to low health
        if self.health < ROBOT_MAX_HEALTH * 0.25 and self.state not in [AIState.RETREAT, AIState.RELOAD]:
            # Retreat to base or nearest teammate
            if nearest_allied_robot:
                self.state = AIState.RETREAT
                self.target = nearest_allied_robot.position
            elif game_state.teams[self.team].base_position:
                self.state = AIState.RETREAT
                self.target = game_state.teams[self.team].base_position
            return
        
        # Check if we're in a position to capture an objective
        if nearest_objective:
            if nearest_objective.type == ObjectiveType.FLAG_CAPTURE:
                if (game_state.objectives[nearest_objective.id].captured_by is None or 
                    game_state.objectives[nearest_objective.id].captured_by == self.team):
                    self.state = AIState.CAPTURE
                    self.target = nearest_objective.position
                    return
            elif nearest_objective.type == ObjectiveType.PAYLOAD_ESCORT:
                self.state = AIState.ESCORT
                self.target = nearest_objective.position
                return
            elif nearest_objective.type == ObjectiveType.CONTROL_POINT:
                if game_state.objectives[nearest_objective.id].captured_by != self.team:
                    self.state = AIState.CAPTURE
                    self.target = nearest_objective.position
                    return
            elif nearest_objective.type == ObjectiveType.BASE_DEFENSE:
                # If base is under attack, defend
                base = game_state.teams[self.team].base_position
                if base and self.position.distance_to(base) < 300:
                    self.state = AIState.DEFEND
                    self.target = base
                    return
        
        # If we're defending and enemy approaches, attack
        if self.state == AIState.DEFEND and nearest_enemy:
            if self.position.distance_to(nearest_enemy.position) < 200:
                self.state = AIState.ATTACK
                self.target = nearest_enemy.position
                return
        
        # If we're escorting and payload is in danger, defend it
        if self.state == AIState.ESCORT and nearest_enemy:
            if self.position.distance_to(nearest_enemy.position) < 250:
                self.state = AIState.ATTACK
                self.target = nearest_enemy.position
                return
        
        # If we're capturing and enemy approaches, attack
        if self.state == AIState.CAPTURE and nearest_enemy:
            if self.position.distance_to(nearest_enemy.position) < 150:
                self.state = AIState.ATTACK
                self.target = nearest_enemy.position
                return
        
        # Check for nearby enemies to attack
        if nearest_enemy and self.position.distance_to(nearest_enemy.position) <= self.vision_range:
            # Determine if we should attack or wait for backup
            nearby_allies = sum(1 for r in game_state.robots if 
                               r.team == self.team and 
                               r.position.distance_to(self.position) < 150 and
                               r.state != AIState.RETREAT)
            
            # Adaptive difficulty: if AI is performing poorly, be more aggressive
            if self.experience < 50 and len(game_state.robots) > 5:
                # Lower experienced bots attack more aggressively
                self.state = AIState.ATTACK
                self.target = nearest_enemy.position
                return
            
            # Higher experienced bots coordinate better
            if nearby_allies >= 2:
                # We have backup, attack
                self.state = AIState.ATTACK
                self.target = nearest_enemy.position
                return
            else:
                # Call for backup or retreat
                self.call_for_help(game_state)
                self.state = AIState.PATROL
                self.target = self.position + Vector2D(50, 50)  # Patrol point
                return
        
        # If no immediate threat, patrol or pursue objective
        if nearest_objective:
            # If objective is not being contested, capture or escort
            if self.state != AIState.CAPTURE and self.state != AIState.ESCORT:
                self.state = AIState.PATROL
                self.target = nearest_objective.position
        else:
            # Patrol around team base
            base = game_state.teams[self.team].base_position
            if base:
                self.state = AIState.PATROL
                # Move in a circular pattern around base
                angle = (time.time() % 10) * 0.6  # Rotate every 10 seconds
                offset = Vector2D(math.cos(angle) * 100, math.sin(angle) * 100)
                self.target = base + offset
    
    def call_for_help(self, game_state: 'GameState') -> None:
        """Send a distress signal to nearby allies."""
        # This would be a message sent to the game state
        # For now, we just flag it for other robots to respond
        for robot in game_state.robots:
            if robot.team == self.team and robot.id != self.id:
                # In a real system, this would update robot's target
                if robot.state == AIState.IDLE or robot.state == AIState.PATROL:
                    # This robot can respond to the call
                    pass
    
    def find_nearest_enemy(self, game_state: 'GameState') -> Optional['Robot']:
        """Find the nearest enemy robot."""
        nearest = None
        min_distance = float('inf')
        
        for robot in game_state.robots:
            if robot.team != self.team and robot.health > 0:
                distance = self.position.distance_to(robot.position)
                if distance < min_distance and distance <= self.vision_range:
                    min_distance = distance
                    nearest = robot
        
        return nearest
    
    def find_nearest_objective(self, game_state: 'GameState') -> Optional['Objective']:
        """Find the nearest objective relevant to the robot's role."""
        nearest = None
        min_distance = float('inf')
        
        for obj_id, objective in game_state.objectives.items():
            # Skip objectives that don't match team goals
            if objective.type == ObjectiveType.BASE_DEFENSE and objective.team == self.team:
                # We're defending our own base - don't pursue
                continue
            elif objective.type == ObjectiveType.FLAG_CAPTURE and objective.team != self.team:
                # We're trying to capture enemy flag
                pass
            elif objective.type == ObjectiveType.PAYLOAD_ESCORT:
                # Only robots assigned to escort will pursue
                if self.role != "escort":
                    continue
            elif objective.type == ObjectiveType.CONTROL_POINT:
                # We want to control points we don't own
                if objective.captured_by == self.team:
                    continue
            
            distance = self.position.distance_to(objective.position)
            if distance < min_distance:
                min_distance = distance
                nearest = objective
        
        return nearest
    
    def find_nearest_teammate(self, game_state: 'GameState') -> Optional['Robot']:
        """Find the nearest teammate."""
        nearest = None
        min_distance = float('inf')
        
        for robot in game_state.robots:
            if robot.team == self.team and robot.id != self.id and robot.health > 0:
                distance = self.position.distance_to(robot.position)
                if distance < min_distance:
                    min_distance = distance
                    nearest = robot
        
        return nearest
    
    def to_dict(self):
        """Serialize robot state for logging or saving."""
        return {
            "id": self.id,
            "team": self.team,
            "position": (self.position.x, self.position.y),
            "health": self.health,
            "energy": self.energy,
            "weapon": self.weapon,
            "weapon_ammo": self.weapon_ammo,
            "weapon_cooldown": self.weapon_cooldown,
            "state": self.state.value,
            "role": self.role,
            "target": (self.target.x, self.target.y) if self.target else None,
            "last_seen_enemy": (self.last_seen_enemy.x, self.last_seen_enemy.y) if self.last_seen_enemy else None,
            "vision_range": self.vision_range,
            "power_ups": list(self.power_ups.keys()),
            "armor": self.armor,
            "experience": self.experience,
            "kill_count": self.kill_count,
            "assist_count": self.assist_count,
            "objectives_completed": self.objectives_completed
        }

@dataclass
class Objective:
    """Represents a game objective (flag, base, payload, etc.)."""
    id: int
    type: ObjectiveType
    position: Vector2D
    team: int = 0  # Team that owns this objective
    captured_by: Optional[int] = None  # Team that currently controls it
    capture_progress: float = 0.0  # For capture objectives (0-100%)
    capture_rate: float = 0.2  # % per second
    radius: float = 40.0
    is_active: bool = True
    
    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type.value,
            "position": (self.position.x, self.position.y),
            "team": self.team,
            "captured_by": self.captured_by,
            "capture_progress": self.capture_progress,
            "capture_rate": self.capture_rate,
            "radius": self.radius,
            "is_active": self.is_active
        }

@dataclass
class PowerUp:
    """Represents a collectible power-up item."""
    id: int
    type: str
    position: Vector2D
    spawn_time: float = field(default_factory=time.time)
    duration: float = 30.0  # How long it stays on ground before despawning
    
    def is_expired(self, current_time: float) -> bool:
        return current_time - self.spawn_time > self.duration
    
    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "position": (self.position.x, self.position.y),
            "spawn_time": self.spawn_time,
            "duration": self.duration
        }

@dataclass
class Hazard:
    """Represents an environmental hazard."""
    id: int
    type: str
    position: Vector2D
    radius: float
    damage_per_second: float
    spawn_time: float = field(default_factory=time.time)
    duration: float = 60.0  # How long the hazard lasts
    
    def is_expired(self, current_time: float) -> bool:
        return current_time - self.spawn_time > self.duration
    
    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "position": (self.position.x, self.position.y),
            "radius": self.radius,
            "damage_per_second": self.damage_per_second,
            "spawn_time": self.spawn_time,
            "duration": self.duration
        }

@dataclass
class Projectile:
    """Represents a fired projectile (bullet, rocket, etc.)."""
    id: int
    origin_robot_id: int
    position: Vector2D
    target: Vector2D
    damage: float
    team: int
    speed: float = 15.0
    range: float = 1000.0
    lifetime: float = 5.0
    lifetime_remaining: float = 5.0
    type: str = "bullet"
    
    def update(self, delta_time: float) -> bool:
        """Update projectile position. Returns False if it should be removed."""
        # Move towards target
        direction = (self.target - self.position).normalize()
        self.position += direction * self.speed * delta_time
        
        # Reduce lifetime
        self.lifetime_remaining -= delta_time
        if self.lifetime_remaining <= 0:
            return False
        
        # Check if out of range
        if self.position.distance_to(self.target) > self.range:
            return False
        
        return True
    
    def to_dict(self):
        return {
            "id": self.id,
            "origin_robot_id": self.origin_robot_id,
            "position": (self.position.x, self.position.y),
            "target": (self.target.x, self.target.y),
            "damage": self.damage,
            "team": self.team,
            "speed": self.speed,
            "range": self.range,
            "lifetime_remaining": self.lifetime_remaining,
            "type": self.type
        }

@dataclass
class Team:
    """Represents a team in the game."""
    id: int
    name: str
    color: Tuple[int, int, int]
    base_position: Vector2D
    score: int = 0
    robots: List[Robot] = field(default_factory=list)
    objectives: List[int] = field(default_factory=list)  # List of objective IDs they control
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "base_position": (self.base_position.x, self.base_position.y),
            "score": self.score,
            "robot_count": len(self.robots),
            "objectives": self.objectives
        }

@dataclass
class GameState:
    """Holds the entire state of the game."""
    current_time: float = 0.0
    teams: Dict[int, Team] = field(default_factory=dict)
    robots: List[Robot] = field(default_factory=list)
    objectives: Dict[int, Objective] = field(default_factory=dict)
    power_ups: List[PowerUp] = field(default_factory=list)
    hazards: List[Hazard] = field(default_factory=list)
    projectiles: List[Projectile] = field(default_factory=list)
    defeated_robots: List[Tuple[int, float]] = field(default_factory=list)  # (robot_id, time_defeated)
    round_start_time: float = field(default_factory=time.time)
    current_objective: Optional[ObjectiveType] = None
    objectives_completed: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    difficulty_level: float = 0.5  # 0.0 = easy, 1.0 = hard
    game_mode: str = "capture_the_flag"  # or "base_defense", "payload_escort"
    is_game_over: bool = False
    winner: Optional[int] = None
    
    def initialize(self) -> None:
        """Initialize the game state with default settings."""
        # Create teams
        for i in range(TEAM_COUNT):
            base_x = 150 if i == 0 else WORLD_WIDTH - 150
            base_y = WORLD_HEIGHT // 2
            team = Team(id=i, name=TEAM_NAMES[i], color=TEAM_COLORS[i], 
                       base_position=Vector2D(base_x, base_y))
            self.teams[i] = team
            
            # Create robots for each team
            robot_count = MAX_ROBOTS_PER_TEAM
            for j in range(robot_count):
                # Distribute robots around base
                angle = (j / robot_count) * 2 * math.pi
                offset = Vector2D(math.cos(angle) * 50, math.sin(angle) * 50)
                position = team.base_position + offset
                
                # Assign roles based on position
                role = "scout" if j == 0 else "attacker" if j < 3 else "defender"
                
                robot = Robot(
                    id=len(self.robots),
                    team=i,
                    position=position,
                    role=role,
                    weapon="pulse_rifle"  # Default weapon
                )
                self.robots.append(robot)
                team.robots.append(robot)
        
        # Create objectives
        self.create_objectives()
        
        # Create initial power-ups
        self.generate_power_ups()
        
        # Create initial hazards
        self.generate_hazards()
        
        # Set game mode
        self.game_mode = random.choice(["capture_the_flag", "base_defense", "payload_escort"])
        self.current_objective = {
            "capture_the_flag": ObjectiveType.FLAG_CAPTURE,
            "base_defense": ObjectiveType.BASE_DEFENSE,
            "payload_escort": ObjectiveType.PAYLOAD_ESCORT
        }[self.game_mode]
        
        # Initialize score
        for team in self.teams.values():
            team.score = 0
        self.objectives_completed = {0: 0, 1: 0}
        self.is_game_over = False
        self.winner = None
    
    def create_objectives(self) -> None:
        """Create objectives based on game mode."""
        # Clear existing objectives
        self.objectives.clear()
        
        if self.game_mode == "capture_the_flag":
            # Create two flags, one for each team
            flag1 = Objective(
                id=0,
                type=ObjectiveType.FLAG_CAPTURE,
                position=Vector2D(100, 200),
                team=0,
                radius=30
            )
            flag2 = Objective(
                id=1,
                type=ObjectiveType.FLAG_CAPTURE,
                position=Vector2D(WORLD_WIDTH - 100, WORLD_HEIGHT - 200),
                team=1,
                radius=30
            )
            self.objectives[0] = flag1
            self.objectives[1] = flag2
            
            # Initial capture status
            flag1.captured_by = 0
            flag2.captured_by = 1
            
        elif self.game_mode == "base_defense":
            # Create a central control point
            control_point = Objective(
                id=0,
                type=ObjectiveType.CONTROL_POINT,
                position=Vector2D(WORLD_WIDTH // 2, WORLD_HEIGHT // 2),
                team=0,
                radius=60
            )
            self.objectives[0] = control_point
            
        elif self.game_mode == "payload_escort":
            # Create a payload and two points
            payload = Objective(
                id=0,
                type=ObjectiveType.PAYLOAD_ESCORT,
                position=Vector2D(100, WORLD_HEIGHT // 2),
                team=0,
                radius=40
            )
            goal = Objective(
                id=1,
                type=ObjectiveType.CONTROL_POINT,
                position=Vector2D(WORLD_WIDTH - 100, WORLD_HEIGHT // 2),
                team=1,
                radius=50
            )
            self.objectives[0] = payload
            self.objectives[1] = goal
    
    def generate_power_ups(self) -> None:
        """Generate initial power-ups."""
        self.power_ups.clear()
        
        # Create 5 random power-ups
        for i in range(5):
            power_up_type = random.choice(list(POWER_UP_TYPES.keys()))
            x = random.randint(100, WORLD_WIDTH - 100)
            y = random.randint(100, WORLD_HEIGHT - 100)
            power_up = PowerUp(
                id=i,
                type=power_up_type,
                position=Vector2D(x, y)
            )
            self.power_ups.append(power_up)
    
    def generate_hazards(self) -> None:
        """Generate initial hazards."""
        self.hazards.clear()
        
        # Create 3 hazards
        for i in range(3):
            hazard_type = random.choice(list(HAZARD_TYPES.keys()))
            x = random.randint(150, WORLD_WIDTH - 150)
            y = random.randint(150, WORLD_HEIGHT - 150)
            hazard = Hazard(
                id=i,
                type=hazard_type,
                position=Vector2D(x, y),
                radius=HAZARD_TYPES[hazard_type]["radius"],
                damage_per_second=HAZARD_TYPES[hazard_type]["damage_per_second"]
            )
            self.hazards.append(hazard)
    
    def add_projectile(self, robot_id: int, origin: Vector2D, target: Vector2D, damage: float, team: int) -> None:
        """Add a new projectile to the game."""
        projectile = Projectile(
            id=len(self.projectiles),
            origin_robot_id=robot_id,
            position=origin,
            target=target,
            damage=damage,
            team=team
        )
        self.projectiles.append(projectile)
    
    def update(self, delta_time: float) -> None:
        """Update the entire game state."""
        self.current_time += delta_time
        
        # Update robots
        for robot in self.robots:
            if robot.health > 0:
                robot.get_ai_decision(self)
                robot.update(delta_time, self)
                
                # If robot is attacking, try to fire
                if robot.state == AIState.ATTACK and robot.target and robot.weapon_cooldown <= 0:
                    # Try to shoot at target if within range
                    if robot.position.distance_to(robot.target) <= WEAPON_TYPES[robot.weapon]["range"]:
                        robot.use_weapon(robot.target, self)
        
        # Update projectiles
        active_projectiles = []
        for projectile in self.projectiles:
            if projectile.update(delta_time):
                active_projectiles.append(projectile)
                # Check for collision with robots
                for robot in self.robots:
                    if robot.team != projectile.team and robot.health > 0:for robot in self.robots:
                    if robot.team != projectile.team and robot.health > 0:
                        weapon = WEAPON_TYPES[self.robots[projectile.origin_robot_id].weapon]
                        projectile_radius = 10 if weapon["type"] == "ranged" else 20
                        if projectile.position.distance_to(robot.position) <= projectile_radius:
                            # Hit detected
                            robot.take_damage(projectile.damage, f"projectile_{projectile.id}")
                            # Add assist to shooter's teammates
                            shooter = next((r for r in self.robots if r.id == projectile.origin_robot_id), None)
                            if shooter:
                                for teammate in self.robots:
                                    if teammate.team == shooter.team and teammate.id != shooter.id and teammate.health > 0:
                                        if teammate.position.distance_to(robot.position) < 150:
                                            teammate.assist_count += 1
                            break  # Projectile hits one target            # Remove projectiles that are out of bounds
            elif (projectile.position.x < 0 or projectile.position.x > WORLD_WIDTH or
                  projectile.position.y < 0 or projectile.position.y > WORLD_HEIGHT):
                # Out of bounds, don't add back
                pass
            else:
                # Lifetime expired or out of range
                pass
        
        self.projectiles = active_projectiles
        
        # Update objectives
        for objective in self.objectives.values():
            if objective.type == ObjectiveType.FLAG_CAPTURE and objective.captured_by is not None:
                # If flag is captured, check for capture progress
                if objective.captured_by == objective.team:
                    # Flag is at home, reset capture progress
                    objective.capture_progress = 0
                else:
                    # Flag is at enemy base, check if it's being returned
                    if objective.captured_by != objective.team:
                        # Capture progress increases based on proximity to home base
                        home_base = self.teams[objective.team].base_position
                        distance_to_home = objective.position.distance_to(home_base)
                        # Higher distance means slower capture progress
                        capture_rate = objective.capture_rate * (1 - distance_to_home / 800)
                        objective.capture_progress += capture_rate * delta_time
                        
                        # If capture progress reaches 100%, flag is captured
                        if objective.capture_progress >= 100:
                            objective.captured_by = objective.team
                            objective.capture_progress = 0
                            # Award points to team
                            self.teams[objective.team].score += 10
                            self.objectives_completed[objective.team] += 1
                            
                            # Reset flag to original position
                            objective.position = Vector2D(100 if objective.team == 0 else WORLD_WIDTH - 100, 
                                                         200 if objective.team == 0 else WORLD_HEIGHT - 200)
            
            elif objective.type == ObjectiveType.CONTROL_POINT:
                # Calculate capture percentage based on number of robots near it
                near_robots = {"team0": 0, "team1": 0}
                for robot in self.robots:
                    if robot.health > 0 and robot.position.distance_to(objective.position) < objective.radius:
                        if robot.team == 0:
                            near_robots["team0"] += 1
                        else:
                            near_robots["team1"] += 1
                
                # Find dominant team
                if near_robots["team0"] > near_robots["team1"]:
                    objective.captured_by = 0
                    objective.capture_progress = min(100, objective.capture_progress + 0.5 * delta_time)
                elif near_robots["team1"] > near_robots["team0"]:
                    objective.captured_by = 1
                    objective.capture_progress = min(100, objective.capture_progress + 0.5 * delta_time)
                else:
                    objective.captured_by = None
                    objective.capture_progress = max(0, objective.capture_progress - 0.3 * delta_time)
                
                # If captured by a team, award score
                if objective.captured_by is not None and objective.capture_progress >= 100:
                    self.teams[objective.captured_by].score += 1
                    self.objectives_completed[objective.captured_by] += 1
                    objective.capture_progress = 0
            
            elif objective.type == ObjectiveType.PAYLOAD_ESCORT:
                # Payload escort - check if payload is near goal
                goal_obj = self.objectives.get(1)
                if goal_obj and objective.position.distance_to(goal_obj.position) < 50:
                    # Payload reached goal
                    objective.capture_progress += 0.5 * delta_time
                    if objective.capture_progress >= 100:
                        # Payload escorted successfully
                        self.teams[objective.team].score += 20
                        self.objectives_completed[objective.team] += 1
                        # Reset payload
                        objective.position = Vector2D(100, WORLD_HEIGHT // 2)
                        objective.capture_progress = 0
        
        # Update power-ups
        self.power_ups = [pu for pu in self.power_ups if not pu.is_expired(self.current_time)]
        
        # Check if any robot has picked up a power-up
        for robot in self.robots:
            if robot.health <= 0:
                continue
            for power_up in self.power_ups[:]:
                if robot.pick_up_power_up(power_up):
                    self.power_ups.remove(power_up)
                    # Spawn a new power-up after a delay
                    if len(self.power_ups) < 5:
                        # Schedule a new power-up to spawn after 10 seconds
                        threading.Timer(10.0, self.spawn_new_power_up).start()
        
        # Update hazards
        self.hazards = [h for h in self.hazards if not h.is_expired(self.current_time)]
        
        # Generate new hazards occasionally
        if len(self.hazards) < 3 and random.random() < 0.01:
            self.generate_hazards()
        
        # Handle respawn of defeated robots
        current_time = self.current_time
        for robot_id, defeat_time in self.defeated_robots[:]:
            respawn_time = defeat_time + 10.0  # 10-second respawn timer
            if current_time >= respawn_time:
                # Find the robot and respawn it at base
                for robot in self.robots:
                    if robot.id == robot_id:
                        team = self.teams[robot.team]
                        robot.position = team.base_position + Vector2D(random.randint(-50, 50), random.randint(-50, 50))
                        robot.health = ROBOT_MAX_HEALTH
                        robot.energy = ROBOT_MAX_ENERGY
                        robot.weapon_ammo = WEAPON_TYPES[robot.weapon]["ammo"]
                        robot.state = AIState.PATROL
                        robot.target = None
                        robot.experience += 2.0  # Bonus for surviving
                        # Remove from defeated list
                        self.defeated_robots.remove((robot_id, defeat_time))
                        break
        
        # Check game over conditions
        self.check_game_over()
        
        # Adaptive difficulty adjustment
        self.adjust_difficulty()
    
    def spawn_new_power_up(self) -> None:
        """Spawn a new power-up after one has been collected."""
        if len(self.power_ups) < 5:  # Don't exceed max
            power_up_type = random.choice(list(POWER_UP_TYPES.keys()))
            x = random.randint(100, WORLD_WIDTH - 100)
            y = random.randint(100, WORLD_HEIGHT - 100)
            power_up = PowerUp(
                id=len(self.power_ups),
                type=power_up_type,
                position=Vector2D(x, y)
            )
            self.power_ups.append(power_up)
    
    def check_game_over(self) -> None:
        """Check if the game has ended."""
        # Check if any team reached the score limit
        SCORE_LIMIT = 50
        for team in self.teams.values():
            if team.score >= SCORE_LIMIT:
                self.is_game_over = True
                self.winner = team.id
                return
        
        # Check if all robots on a team are defeated
        for team_id, team in self.teams.items():
            alive_robots = sum(1 for r in self.robots if r.team == team_id and r.health > 0)
            if alive_robots == 0:
                # Team eliminated
                self.is_game_over = True
                # Winner is the other team
                self.winner = 1 if team_id == 0 else 0
                return
    
    def adjust_difficulty(self) -> None:
        """Adjust AI difficulty based on game performance."""
        # Calculate team success rates
        team_success = {}
        for team_id in [0, 1]:
            team = self.teams[team_id]
            total_robots = len(team.robots)
            alive_robots = sum(1 for r in self.robots if r.team == team_id and r.health > 0)
            objectives_completed = self.objectives_completed[team_id]
            
            # Success rate based on objectives completed and robots alive
            success_rate = (objectives_completed / (self.current_time / 60 + 1)) * 0.5 + (alive_robots / total_robots) * 0.5
            
            # If team is performing very well, increase difficulty
            if success_rate > 0.7:
                self.difficulty_level = min(1.0, self.difficulty_level + 0.001)
            elif success_rate < 0.3:
                self.difficulty_level = max(0.0, self.difficulty_level - 0.001)
        
        # Apply difficulty to AI behavior
        for robot in self.robots:
            if robot.team != 0:  # Only apply to AI-controlled bots (team 1)
                # Increase aggression when difficulty is high
                if self.difficulty_level > 0.7:
                    robot.vision_range = min(300, robot.vision_range + 0.5)
                    if robot.role == "scout":
                        robot.role = "attacker"
                elif self.difficulty_level < 0.3:
                    robot.vision_range = max(100, robot.vision_range - 0.5)
                    if robot.role == "attacker":
                        robot.role = "scout"
    
    def to_dict(self) -> Dict:
        """Serialize entire game state for debugging or saving."""
        return {
            "current_time": self.current_time,
            "teams": {team_id: team.to_dict() for team_id, team in self.teams.items()},
            "robots": [robot.to_dict() for robot in self.robots],
            "objectives": {obj_id: obj.to_dict() for obj_id, obj in self.objectives.items()},
            "power_ups": [pu.to_dict() for pu in self.power_ups],
            "hazards": [h.to_dict() for h in self.hazards],
            "projectiles": [p.to_dict() for p in self.projectiles],
            "defeated_robots": self.defeated_robots,
            "round_start_time": self.round_start_time,
            "current_objective": self.current_objective.value if self.current_objective else None,
            "objectives_completed": self.objectives_completed,
            "difficulty_level": self.difficulty_level,
            "game_mode": self.game_mode,
            "is_game_over": self.is_game_over,
            "winner": self.winner
        }

# ============================
# MAIN GAME CLASS
# ============================

class CyberArena:
    """Main game controller for CyberArena."""
    
    def __init__(self):
        self.game_state = GameState()
        self.game_state.initialize()
        
        # Game frame rate control
        self.last_update = time.time()
        self.fps = 60
        self.delta_time = 1.0 / self.fps
        
        # Simulation state
        self.is_running = True
        self.paused = False
        
        # Performance monitoring
        self.frame_times = deque(maxlen=100)
        
        # Learning system - store past battles for adaptation
        self.battle_history = []
        self.learning_model = {}
    
    def run(self) -> None:
        """Main game loop."""
        print("CyberArena: Futuristic Robot Battle Arena Started")
        print(f"Game Mode: {self.game_state.game_mode}")
        print("Press 'q' to quit, 'p' to pause, 's' to save state\n")
        
        while self.is_running:
            current_time = time.time()
            elapsed = current_time - self.last_update
            
            # Cap frame rate
            if elapsed < self.delta_time:
                time.sleep(self.delta_time - elapsed)
                continue
            
            self.delta_time = elapsed
            self.last_update = current_time
            
            # Track performance
            self.frame_times.append(elapsed)
            
            # Update game state
            if not self.paused:
                self.game_state.update(self.delta_time)
            
            # Render state (in a real implementation, this would be graphics)
            self.render()
            
            # Check for input (simulated)
            self.handle_input()
            
            # Check for game over and restart if needed
            if self.game_state.is_game_over:
                self.handle_game_over()
    
    def render(self) -> None:
        """Render the current state (text-based simulation)."""
        # Print game status
        print(f"\n{'='*80}")
        print(f"Time: {self.game_state.current_time:.1f}s | FPS: {1/self.delta_time:.1f}")
        print(f"Game Mode: {self.game_state.game_mode} | Difficulty: {self.game_state.difficulty_level:.2f}")
        print(f"Team Scores: {self.game_state.teams[0].name}: {self.game_state.teams[0].score} | "
              f"{self.game_state.teams[1].name}: {self.game_state.teams[1].score}")
        
        # Print objectives status
        for obj_id, obj in self.game_state.objectives.items():
            status = f"Team {obj.team}" if obj.captured_by is not None else "Neutral"
            print(f"Objective {obj_id} ({obj.type.value}): {status} | "
                  f"Progress: {obj.capture_progress:.1f}%")
        
        # Print robot status
        for team_id, team in self.game_state.teams.items():
            print(f"\n{team.name} - {len(team.robots)} robots | Score: {team.score}")
            for robot in team.robots:
                if robot.health > 0:
                    weapon_info = WEAPON_TYPES[robot.weapon]
                    ammo_str = f"{robot.weapon_ammo}/{weapon_info['ammo']}" if weapon_info['ammo'] > 0 else "∞"
                    power_str = ", ".join(robot.power_ups.keys()) if robot.power_ups else "None"
                    print(f"  Robot {robot.id}: HP:{robot.health:.0f} | "
                          f"EN:{robot.energy:.0f} | "
                          f"Weapon: {robot.weapon}({ammo_str}) | "
                          f"State: {robot.state.value} | "
                          f"Role: {robot.role} | "
                          f"XP:{robot.experience:.1f} | "
                          f"Kills:{robot.kill_count} | "
                          f"Assists:{robot.assist_count} | "
                          f"Power-ups: {power_str}")
        
        # Print power-ups and hazards
        print(f"\nPower-ups: {len(self.game_state.power_ups)} | "
              f"Hazards: {len(self.game_state.hazards)} | "
              f"Projectiles: {len(self.game_state.projectiles)}")
        
        # Check if game over
        if self.game_state.is_game_over:
            winner_name = self.game_state.teams[self.game_state.winner].name
            print(f"\n🎉 GAME OVER - {winner_name} WINS! 🎉")
    
    def handle_input(self) -> None:
        """Handle simulated user input."""
        # In a real implementation, this would be keyboard/mouse input
        # For simulation, we'll just exit after a certain time or on command
        if time.time() - self.game_state.round_start_time > 180:  # 3-minute limit
            self.is_running = False
            print("\nTime limit reached. Ending simulation.")
            return
        
        # Simulate user input (for testing)
        if random.random() < 0.01:  # 1% chance per frame to simulate quit
            self.is_running = False
            print("\nSimulated quit.")
    
    def handle_game_over(self) -> None:
        """Handle game over conditions."""
        # Save battle data for learning
        battle_data = {
            "timestamp": time.time(),
            "game_mode": self.game_state.game_mode,
            "difficulty": self.game_state.difficulty_level,
            "winner": self.game_state.winner,
            "team_scores": {team_id: team.score for team_id, team in self.game_state.teams.items()},
            "robot_stats": [robot.to_dict() for robot in self.game_state.robots],
            "objectives_completed": self.game_state.objectives_completed.copy()
        }
        self.battle_history.append(battle_data)
        
        # Save battle data to file
        self.save_battle_history()
        
        # Reset game after a delay
        print(f"\nReseting game in 5 seconds...")
        time.sleep(5)
        
        # Reset game state
        self.game_state = GameState()
        self.game_state.initialize()
    
    def save_battle_history(self) -> None:
        """Save battle history to a JSON file."""
        try:
            with open("cyberarena_battle_history.json", "w") as f:
                json.dump(self.battle_history, f, indent=2)
        except Exception as e:
            print(f"Error saving battle history: {e}")

# ============================
# ENTRY POINT
# ============================

if __name__ == "__main__":
    # Create and start the CyberArena simulation
    game = CyberArena()
    game.run()