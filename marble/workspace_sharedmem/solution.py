# solution.py
import math
import random
import time
import threading
import queue
import json
import copy
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

# -----------------------------
# CONFIGURATION AND CONSTANTS
# -----------------------------

# Game constants
GAME_WIDTH = 1000
GAME_HEIGHT = 800
MAX_ROBOTS_PER_TEAM = 5
MAX_TEAMS = 4
MAX_ROBOTS = MAX_TEAMS * MAX_ROBOTS_PER_TEAM
UPDATE_RATE = 60  # FPS
TARGET_UPDATE_INTERVAL = 1.0 / UPDATE_RATE

# Team colors
TEAM_COLORS = [
    (255, 0, 0),    # Red
    (0, 0, 255),    # Blue
    (0, 255, 0),    # Green
    (255, 255, 0),  # Yellow
]

# Weapon types
class WeaponType(Enum):
    RANGED = "ranged"
    MELEE = "melee"
    SUPPORT = "support"

# Objective types
class ObjectiveType(Enum):
    CAPTURE_FLAG = "capture_flag"
    DEFEND_BASE = "defend_base"
    ESCORT_PAYLOAD = "escort_payload"

# Power-up types
class PowerUpType(Enum):
    HEALTH_PACK = "health_pack"
    AMMO_PACK = "ammo_pack"
    SPEED_BOOST = "speed_boost"
    SHIELD = "shield"
    SUPER_WEAPON = "super_weapon"

# Robot roles
class RobotRole(Enum):
    ATTACKER = "attacker"
    DEFENDER = "defender"
    SUPPORT = "support"
    SCOUT = "scout"
    CONTROLLER = "controller"

# Damage types
class DamageType(Enum):
    BALLISTIC = "ballistic"
    ENERGY = "energy"
    EXPLOSIVE = "explosive"
    MELEE = "melee"

# -----------------------------
# DATA STRUCTURES
# -----------------------------

@dataclass
class Position:
    x: float = 0.0
    y: float = 0.0
    
    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def normalize(self) -> 'Position':
        magnitude = math.sqrt(self.x**2 + self.y**2)
        if magnitude == 0:
            return Position(0, 0)
        return Position(self.x / magnitude, self.y / magnitude)
    
    def __add__(self, other: 'Position') -> 'Position':
        return Position(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Position') -> 'Position':
        return Position(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Position':
        return Position(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar: float) -> 'Position':
        if scalar == 0:
            return Position(0, 0)
        return Position(self.x / scalar, self.y / scalar)

@dataclass
class Velocity:
    x: float = 0.0
    y: float = 0.0
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)
    
    def normalize(self) -> 'Velocity':
        mag = self.magnitude()
        if mag == 0:
            return Velocity(0, 0)
        return Velocity(self.x / mag, self.y / mag)
    
    def __add__(self, other: 'Velocity') -> 'Velocity':
        return Velocity(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Velocity') -> 'Velocity':
        return Velocity(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Velocity':
        return Velocity(self.x * scalar, self.y * scalar)

@dataclass
class Weapon:
    type: WeaponType
    name: str
    damage: float
    range: float
    fire_rate: float  # seconds between shots
    ammo: int
    max_ammo: int
    reload_time: float = 2.0
    projectile_speed: float = 1000.0  # pixels per second
    splash_radius: float = 0.0
    damage_type: DamageType = DamageType.BALLISTIC
    is_active: bool = True
    
    def __post_init__(self):
        self.last_fired = 0.0
        self.ammo_left = self.ammo
        self.is_reloading = False
        self.reload_timer = 0.0

@dataclass
class Robot:
    id: int
    team: int
    position: Position
    velocity: Velocity
    health: float
    max_health: float
    armor: float
    speed: float
    role: RobotRole
    weapons: List[Weapon]
    current_weapon_index: int = 0
    energy: float = 100.0
    max_energy: float = 100.0
    cooldown: float = 0.0
    last_action_time: float = 0.0
    vision_range: float = 300.0
    is_alive: bool = True
    is_capturing: bool = False
    capture_progress: float = 0.0
    power_ups: Dict[PowerUpType, float] = field(default_factory=dict)  # power_up_type -> duration
    knowledge_base: Dict[str, Any] = field(default_factory=dict)  # AI memory for learning
    
    def __post_init__(self):
        if not self.weapons:
            # Default weapon if none provided
            self.weapons = [
                Weapon(
                    type=WeaponType.RANGED,
                    name="Plasma Rifle",
                    damage=25.0,
                    range=400.0,
                    fire_rate=0.5,
                    ammo=30,
                    max_ammo=60,
                    projectile_speed=800.0
                )
            ]
        self.current_weapon = self.weapons[self.current_weapon_index]
        
    def get_current_weapon(self) -> Weapon:
        return self.weapons[self.current_weapon_index]
    
    def has_weapon(self, weapon_type: WeaponType) -> bool:
        return any(w.type == weapon_type for w in self.weapons)
    
    def can_fire(self) -> bool:
        weapon = self.get_current_weapon()
        if not weapon.is_active:
            return False
        if weapon.is_reloading:
            return False
        if weapon.ammo_left <= 0:
            return False
        current_time = time.time()
        return (current_time - weapon.last_fired) >= weapon.fire_rate
    
    def fire_weapon(self) -> Optional['Projectile']:
        if not self.can_fire():
            return None
            
        weapon = self.get_current_weapon()
        weapon.last_fired = time.time()
        weapon.ammo_left -= 1
        
        # Calculate direction based on target or movement
        # For simplicity, we'll assume the robot is aiming at its current target
        # In a real implementation, this would be more sophisticated
        direction = Position(1, 0)  # default right
        
        # Create projectile
        projectile = Projectile(
            position=Position(self.position.x, self.position.y),
            velocity=Velocity(direction.x * weapon.projectile_speed, direction.y * weapon.projectile_speed),
            damage=weapon.damage,
            damage_type=weapon.damage_type,
            range=weapon.range,
            splash_radius=weapon.splash_radius,
            owner_id=self.id,
            team=self.team
        )
        
        return projectile
    
    def reload_weapon(self) -> bool:
        weapon = self.get_current_weapon()
        if weapon.is_reloading:
            weapon.reload_timer += TARGET_UPDATE_INTERVAL
            if weapon.reload_timer >= weapon.reload_time:
                weapon.ammo_left = min(weapon.max_ammo, weapon.ammo_left + 15)  # Reload 15 ammo
                weapon.is_reloading = False
                weapon.reload_timer = 0.0
                return True
        elif weapon.ammo_left < weapon.max_ammo:
            weapon.is_reloading = True
            weapon.reload_timer = 0.0
        return False
    
    def use_power_up(self, power_up_type: PowerUpType) -> bool:
        if power_up_type in self.power_ups:
            # Apply power-up effect
            if power_up_type == PowerUpType.HEALTH_PACK:
                self.health = min(self.max_health, self.health + 30)
            elif power_up_type == PowerUpType.AMMO_PACK:
                weapon = self.get_current_weapon()
                weapon.ammo_left = min(weapon.max_ammo, weapon.ammo_left + 20)
            elif power_up_type == PowerUpType.SPEED_BOOST:
                self.speed *= 1.5
                self.power_ups[power_up_type] = 5.0  # 5 seconds duration
            elif power_up_type == PowerUpType.SHIELD:
                self.armor += 50
                self.power_ups[power_up_type] = 8.0
            elif power_up_type == PowerUpType.SUPER_WEAPON:
                # Temporarily upgrade weapon
                weapon = self.get_current_weapon()
                weapon.damage *= 2
                weapon.fire_rate *= 0.5
                self.power_ups[power_up_type] = 10.0
            return True
        return False
    
    def update_power_ups(self, delta_time: float) -> None:
        to_remove = []
        for power_type, duration in self.power_ups.items():
            self.power_ups[power_type] = duration - delta_time
            if self.power_ups[power_type] <= 0:
                to_remove.append(power_type)
        
        for power_type in to_remove:
            # Remove the effect
            if power_type == PowerUpType.SPEED_BOOST:
                self.speed /= 1.5
            elif power_type == PowerUpType.SHIELD:
                self.armor = max(0, self.armor - 50)
            elif power_type == PowerUpType.SUPER_WEAPON:
                weapon = self.get_current_weapon()
                weapon.damage /= 2
                weapon.fire_rate /= 0.5
            del self.power_ups[power_type]

@dataclass
class Projectile:
    position: Position
    velocity: Velocity
    damage: float
    damage_type: DamageType
    range: float
    splash_radius: float
    owner_id: int
    team: int
    lifetime: float = 5.0  # seconds
    age: float = 0.0
    is_active: bool = True
    
    def update(self, delta_time: float) -> None:
        self.position = self.position + (self.velocity * delta_time)
        self.age += delta_time
        if self.age >= self.lifetime:
            self.is_active = False

@dataclass
class PowerUp:
    position: Position
    type: PowerUpType
    spawn_time: float = 0.0
    duration: float = 30.0  # seconds before despawning
    is_collected: bool = False
    value: float = 1.0  # multiplier for scoring

@dataclass
class Objective:
    type: ObjectiveType
    position: Position
    team: int = -1  # team assigned to objective
    progress: float = 0.0
    max_progress: float = 100.0
    captured_by: Optional[int] = None  # team that captured it
    is_active: bool = True
    radius: float = 50.0
    description: str = ""
    
    def update(self, delta_time: float, active_robots: List[Robot]) -> None:
        if not self.is_active:
            return
            
        # For capture objectives, check nearby robots
        if self.type == ObjectiveType.CAPTURE_FLAG:
            nearby_robots = [
                robot for robot in active_robots 
                if robot.is_alive and robot.position.distance_to(self.position) <= self.radius
            ]
            
            # Check if any robot from the owning team is near
            if nearby_robots:
                team_count = defaultdict(int)
                for robot in nearby_robots:
                    if robot.team == self.team:
                        team_count[robot.team] += 1
                
                # If robot from team is near, increase progress
                if self.team in team_count and team_count[self.team] > 0:
                    self.progress += 0.5 * delta_time * team_count[self.team]
                    if self.progress >= self.max_progress:
                        self.captured_by = self.team
                        self.is_active = False
                        return
                else:
                    # If robots from other teams are near, they can steal
                    for robot in nearby_robots:
                        if robot.team != self.team and self.captured_by == robot.team:
                            # Steal progress
                            self.progress -= 0.3 * delta_time
                            if self.progress <= 0:
                                self.captured_by = None
                                self.progress = 0

@dataclass
class Environment:
    hazards: List[Position] = field(default_factory=list)  # positions of environmental hazards
    hazard_radius: float = 30.0
    hazard_damage: float = 10.0
    hazard_interval: float = 10.0  # seconds between hazard updates
    last_hazard_update: float = 0.0
    lighting: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.lighting = {
            "ambient": 0.3,
            "directional": {
                "intensity": 0.7,
                "direction": Position(0.5, -0.5),  # top-right
                "color": (255, 255, 200)
            },
            "dynamic": []  # list of dynamic light sources
        }
    
    def add_hazard(self, position: Position) -> None:
        self.hazards.append(position)
    
    def update_hazards(self, current_time: float) -> None:
        if current_time - self.last_hazard_update >= self.hazard_interval:
            # Randomly spawn new hazard
            if len(self.hazards) < 8:  # Max 8 hazards
                new_hazard = Position(
                    random.uniform(50, GAME_WIDTH - 50),
                    random.uniform(50, GAME_HEIGHT - 50)
                )
                self.add_hazard(new_hazard)
            self.last_hazard_update = current_time

@dataclass
class GameStats:
    team_scores: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    total_kills: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    total_deaths: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    objectives_completed: Dict[ObjectiveType, int] = field(default_factory=lambda: defaultdict(int))
    time_played: float = 0.0
    last_update_time: float = 0.0
    
    def update(self, delta_time: float) -> None:
        self.time_played += delta_time
        self.last_update_time = time.time()
    
    def add_kill(self, killer_team: int, victim_team: int) -> None:
        self.total_kills[killer_team] += 1
        self.total_deaths[victim_team] += 1
        self.team_scores[killer_team] += 10.0  # Points for kill
    
    def add_objective(self, objective_type: ObjectiveType, team: int) -> None:
        self.objectives_completed[objective_type] += 1
        self.team_scores[team] += 50.0  # Points for objective
    
    def get_leaderboard(self) -> List[Tuple[int, float]]:
        return sorted(self.team_scores.items(), key=lambda x: x[1], reverse=True)

@dataclass
class GameSettings:
    game_mode: str = "team_deathmatch"
    max_duration: float = 300.0  # 5 minutes
    adaptive_difficulty: bool = True
    difficulty_level: float = 0.5  # 0.0 to 1.0
    robot_ai_complexity: float = 0.7  # 0.0 to 1.0
    spawn_points: List[Position] = field(default_factory=list)
    map_layout: str = "default"
    
    def __post_init__(self):
        if not self.spawn_points:
            # Default spawn points for 4 teams
            self.spawn_points = [
                Position(100, 100),      # Team 0
                Position(GAME_WIDTH - 100, 100),  # Team 1
                Position(100, GAME_HEIGHT - 100), # Team 2
                Position(GAME_WIDTH - 100, GAME_HEIGHT - 100), # Team 3
            ]

# -----------------------------
# AI BEHAVIOR SYSTEM
# -----------------------------

class RobotAI:
    """
    AI module that controls robot behavior using state machines and reinforcement learning principles
    """
    
    def __init__(self, robot: Robot, game_state: 'CyberArena'):
        self.robot = robot
        self.game_state = game_state
        self.state = "patrol"  # patrol, attack, defend, capture, flee, support
        self.target: Optional[Robot] = None
        self.objective_target: Optional[Objective] = None
        self.last_thought_time = 0.0
        self.thought_interval = 0.5  # seconds between AI decisions
        
        # Learning variables
        self.reinforcement_history = deque(maxlen=100)  # Store recent outcomes
        self.strategy_weights = {
            "attack": 0.3,
            "defend": 0.3,
            "support": 0.2,
            "capture": 0.2,
            "patrol": 0.1
        }
        
        # Memory of past events
        self.knowledge = {
            "enemy_positions": [],
            "ally_positions": [],
            "successful_actions": 0,
            "failed_actions": 0,
            "team_cooperation_score": 0.5
        }
    
    def update(self, delta_time: float) -> None:
        current_time = time.time()
        
        # Update knowledge base
        self.update_knowledge()
        
        # Only update AI decision at intervals
        if current_time - self.last_thought_time < self.thought_interval:
            return
            
        self.last_thought_time = current_time
        
        # Calculate AI complexity multiplier
        complexity = self.game_state.settings.robot_ai_complexity
        if self.game_state.settings.adaptive_difficulty:
            # Adjust based on performance
            complexity = self.calculate_adaptive_complexity()
        
        # Decision making based on current state and environment
        self.decide_behavior(complexity)
        
        # Update strategy weights based on reinforcement learning
        self.update_strategy_weights()
    
    def update_knowledge(self) -> None:
        """Update robot's knowledge of the environment"""
        # Record enemy and ally positions
        self.knowledge["enemy_positions"] = []
        self.knowledge["ally_positions"] = []
        
        for robot in self.game_state.robots:
            if robot.id == self.robot.id:
                continue
            if robot.team == self.robot.team and robot.is_alive:
                self.knowledge["ally_positions"].append({
                    "id": robot.id,
                    "position": robot.position,
                    "health": robot.health,
                    "role": robot.role
                })
            elif robot.team != self.robot.team and robot.is_alive:
                self.knowledge["enemy_positions"].append({
                    "id": robot.id,
                    "position": robot.position,
                    "health": robot.health,
                    "role": robot.role
                })
    
    def calculate_adaptive_complexity(self) -> float:
        """Calculate adaptive AI difficulty based on team performance"""
        # Get team's performance metrics
        team_score = self.game_state.stats.team_scores[self.robot.team]
        team_kills = self.game_state.stats.total_kills[self.robot.team]
        team_deaths = self.game_state.stats.total_deaths[self.robot.team]
        
        # Calculate win rate (normalized)
        if team_kills + team_deaths == 0:
            win_rate = 0.5
        else:
            win_rate = team_kills / (team_kills + team_deaths)
        
        # If team is doing well, make AI harder
        # If team is struggling, make AI easier
        if win_rate > 0.6:
            return min(1.0, self.game_state.settings.robot_ai_complexity + 0.1)
        elif win_rate < 0.4:
            return max(0.1, self.game_state.settings.robot_ai_complexity - 0.1)
        
        return self.game_state.settings.robot_ai_complexity
    
    def update_strategy_weights(self) -> None:
        """Update strategy weights based on reinforcement learning"""
        if len(self.reinforcement_history) < 5:
            return  # Need more data
            
        # Calculate success rate for each strategy
        strategy_success = defaultdict(list)
        for outcome in self.reinforcement_history:
            strategy = outcome["strategy"]
            success = outcome["success"]
            strategy_success[strategy].append(success)
        
        # Update weights based on success rates
        for strategy in self.strategy_weights:
            if strategy_success[strategy]:
                success_rate = sum(strategy_success[strategy]) / len(strategy_success[strategy])
                # Adjust weight based on success
                self.strategy_weights[strategy] = max(0.05, min(0.8, success_rate * 2))
        
        # Normalize weights
        total = sum(self.strategy_weights.values())
        if total > 0:
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] /= total
    
    def decide_behavior(self, complexity: float) -> None:
        """Decide the robot's next behavior based on current state"""
        if not self.robot.is_alive:
            return
            
        # Get current conditions
        nearest_enemy = self.find_nearest_enemy()
        nearest_ally = self.find_nearest_ally()
        nearest_objective = self.find_nearest_objective()
        health_ratio = self.robot.health / self.robot.max_health
        ammo_ratio = self.robot.get_current_weapon().ammo_left / self.robot.get_current_weapon().max_ammo
        
        # Determine priority based on strategy weights
        strategy_choices = []
        for strategy, weight in self.strategy_weights.items():
            if weight > 0.05:  # Only consider strategies with reasonable weight
                strategy_choices.append((strategy, weight))
        
        # If we have a clear priority, use it
        if nearest_enemy and health_ratio > 0.3:
            if nearest_enemy.position.distance_to(self.robot.position) < 150:
                # Close range - melee or aggressive
                if self.robot.has_weapon(WeaponType.MELEE) and random.random() < complexity:
                    self.state = "attack"
                    self.target = nearest_enemy
                    return
                else:
                    # Use ranged weapon
                    self.state = "attack"
                    self.target = nearest_enemy
                    return
        
        # Capture objective if available and we're not near enemies
        if nearest_objective and nearest_objective.captured_by != self.robot.team:
            distance_to_obj = nearest_objective.position.distance_to(self.robot.position)
            if distance_to_obj < 200 and not nearest_enemy:
                if random.random() < complexity * 0.8:
                    self.state = "capture"
                    self.objective_target = nearest_objective
                    return
        
        # Defend if we're near our base and enemies are nearby
        if nearest_objective and nearest_objective.captured_by == self.robot.team:
            if nearest_enemy and nearest_enemy.position.distance_to(nearest_objective.position) < 100:
                self.state = "defend"
                self.target = nearest_enemy
                return
        
        # Support if allies are low on health
        if nearest_ally and health_ratio > 0.5:
            if nearest_ally.health / nearest_ally.max_health < 0.3:
                if self.robot.has_weapon(WeaponType.SUPPORT) and random.random() < complexity * 0.7:
                    self.state = "support"
                    self.target = nearest_ally
                    return
        
        # Flee if low on health
        if health_ratio < 0.2:
            if nearest_enemy and nearest_enemy.position.distance_to(self.robot.position) < 150:
                self.state = "flee"
                self.target = nearest_enemy
                return
        
        # Default behavior
        self.state = "patrol"
        self.target = None
        self.objective_target = None
    
    def find_nearest_enemy(self) -> Optional[Robot]:
        """Find the nearest enemy robot"""
        if not self.knowledge["enemy_positions"]:
            return None
            
        nearest = None
        min_dist = float('inf')
        
        for enemy in self.knowledge["enemy_positions"]:
            dist = self.robot.position.distance_to(enemy["position"])
            if dist < min_dist:
                min_dist = dist
                nearest = None
                for robot in self.game_state.robots:
                    if robot.id == enemy["id"]:
                        nearest = robot
                        break
        
        return nearest
    
    def find_nearest_ally(self) -> Optional[Robot]:
        """Find the nearest ally robot"""
        if not self.knowledge["ally_positions"]:
            return None
            
        nearest = None
        min_dist = float('inf')
        
        for ally in self.knowledge["ally_positions"]:
            dist = self.robot.position.distance_to(ally["position"])
            if dist < min_dist:
                min_dist = dist
                nearest = None
                for robot in self.game_state.robots:
                    if robot.id == ally["id"]:
                        nearest = robot
                        break
        
        return nearest
    
    def find_nearest_objective(self) -> Optional[Objective]:
        """Find the nearest objective"""
        if not self.game_state.objectives:
            return None
            
        nearest = None
        min_dist = float('inf')
        
        for obj in self.game_state.objectives:
            if not obj.is_active:
                continue
            dist = self.robot.position.distance_to(obj.position)
            if dist < min_dist:
                min_dist = dist
                nearest = obj
        
        return nearest
    
    def get_action(self) -> Dict[str, Any]:
        """Return the action the robot should take based on current state"""
        action = {
            "move_direction": Position(0, 0),
            "fire": False,
            "switch_weapon": False,
            "reload": False,
            "use_power_up": None
        }
        
        if not self.robot.is_alive:
            return action
        
        # Determine movement direction based on state
        if self.state == "attack" and self.target:
            # Move toward target
            direction = (self.target.position - self.robot.position).normalize()
            action["move_direction"] = direction
            
            # Fire if in range and weapon is ready
            if self.robot.position.distance_to(self.target.position) < self.robot.get_current_weapon().range:
                if self.robot.can_fire():
                    action["fire"] = True
                else:
                    action["reload"] = not self.robot.get_current_weapon().is_reloading
        
        elif self.state == "defend" and self.target:
            # Move toward target, but maintain distance
            direction = (self.target.position - self.robot.position).normalize()
            # Move away if too close
            if self.robot.position.distance_to(self.target.position) < 100:
                action["move_direction"] = direction * -1
            else:
                action["move_direction"] = direction
            
            # Fire if we have a weapon and are in range
            if self.robot.position.distance_to(self.target.position) < self.robot.get_current_weapon().range:
                if self.robot.can_fire():
                    action["fire"] = True
        
        elif self.state == "capture" and self.objective_target:
            # Move toward objective
            direction = (self.objective_target.position - self.robot.position).normalize()
            action["move_direction"] = direction
            
            # When close enough, start capturing
            if self.robot.position.distance_to(self.objective_target.position) < self.objective_target.radius:
                self.robot.is_capturing = True
                self.robot.capture_progress = min(100, self.robot.capture_progress + 0.5)
        
        elif self.state == "support" and self.target:
            # Move toward ally
            direction = (self.target.position - self.robot.position).normalize()
            action["move_direction"] = direction
            
            # If we have a support weapon and are close enough
            if self.robot.position.distance_to(self.target.position) < 150:
                if self.robot.has_weapon(WeaponType.SUPPORT):
                    action["fire"] = True
        
        elif self.state == "flee" and self.target:
            # Move away from enemy
            direction = (self.robot.position - self.target.position).normalize()
            action["move_direction"] = direction
        
        else:  # patrol
            # Random patrol behavior
            if random.random() < 0.05:  # 5% chance to change direction
                action["move_direction"] = Position(
                    random.uniform(-1, 1),
                    random.uniform(-1, 1)
                ).normalize()
            else:
                # Continue current movement
                action["move_direction"] = self.robot.velocity.normalize()
        
        # Check if we need to reload
        if self.robot.get_current_weapon().ammo_left <= 5 and not self.robot.get_current_weapon().is_reloading:
            action["reload"] = True
        
        # Switch to a better weapon if available
        if self.robot.has_weapon(WeaponType.MELEE) and self.robot.position.distance_to(self.target.position) < 80:
            for i, weapon in enumerate(self.robot.weapons):
                if weapon.type == WeaponType.MELEE:
                    action["switch_weapon"] = True
                    break
        
        # Use available power-ups
        for power_type in self.robot.power_ups:
            if self.robot.power_ups[power_type] > 0 and random.random() < 0.1:
                action["use_power_up"] = power_type
                break
        
        return action

# -----------------------------
# GAME ENGINE
# -----------------------------

class CyberArena:
    """
    Main game engine for CyberArena - a multi-agent robotic battle system
    """
    
    def __init__(self, settings: GameSettings = None):
        self.settings = settings or GameSettings()
        self.robots: List[Robot] = []
        self.projectiles: List[Projectile] = []
        self.power_ups: List[PowerUp] = []
        self.objectives: List[Objective] = []
        self.environment = Environment()
        self.stats = GameStats()
        self.ai_controllers: Dict[int, RobotAI] = {}  # robot_id -> AI controller
        self.game_running = False
        self.game_start_time = 0.0
        self.last_update_time = 0.0
        self.current_time = 0.0
        self.audio_queue = queue.Queue()  # For audio effects
        self.visual_effects = []  # For particle effects
        
        # Initialize game
        self.initialize_game()
    
    def initialize_game(self) -> None:
        """Initialize the game with robots, objectives, and power-ups"""
        # Create robots
        for team in range(MAX_TEAMS):
            for i in range(MAX_ROBOTS_PER_TEAM):
                robot_id = team * MAX_ROBOTS_PER_TEAM + i
                if robot_id >= MAX_ROBOTS:
                    break
                
                # Spawn position
                spawn_pos = self.settings.spawn_points[team % len(self.settings.spawn_points)]
                
                # Random role distribution
                roles = list(RobotRole)
                role = roles[robot_id % len(roles)]
                
                # Random weapon set
                weapons = []
                
                # Always give a ranged weapon
                weapons.append(
                    Weapon(
                        type=WeaponType.RANGED,
                        name=f"Plasma Rifle-{robot_id}",
                        damage=25.0,
                        range=400.0,
                        fire_rate=0.5,
                        ammo=30,
                        max_ammo=60,
                        projectile_speed=800.0,
                        damage_type=DamageType.ENERGY
                    )
                )
                
                # 70% chance for melee weapon
                if random.random() < 0.7:
                    weapons.append(
                        Weapon(
                            type=WeaponType.MELEE,
                            name=f"Plasma Blade-{robot_id}",
                            damage=45.0,
                            range=70.0,
                            fire_rate=1.5,
                            ammo=0,
                            max_ammo=0,
                            damage_type=DamageType.MELEE
                        )
                    )
                
                # 30% chance for support weapon
                if random.random() < 0.3:
                    weapons.append(
                        Weapon(
                            type=WeaponType.SUPPORT,
                            name=f"Medi-Pulse-{robot_id}",
                            damage=0.0,
                            range=200.0,
                            fire_rate=1.0,
                            ammo=20,
                            max_ammo=20,
                            projectile_speed=500.0,
                            damage_type=DamageType.HEALING
                        )
                    )
                
                # Create robot
                robot = Robot(
                    id=robot_id,
                    team=team,
                    position=spawn_pos,
                    velocity=Velocity(0, 0),
                    health=100.0,
                    max_health=100.0,
                    armor=20.0,
                    speed=150.0,
                    role=role,
                    weapons=weapons
                )
                
                self.robots.append(robot)
                self.ai_controllers[robot_id] = RobotAI(robot, self)
        
        # Create objectives based on game mode
        self.create_objectives()
        
        # Create initial power-ups
        self.spawn_power_ups(5)
        
        # Initialize game clock
        self.game_start_time = time.time()
        self.last_update_time = self.game_start_time
        self.current_time = self.game_start_time
    
    def create_objectives(self) -> None:
        """Create objectives based on game mode"""
        # For different game modes, create different objectives
        if self.settings.game_mode == "capture_the_flag":
            # Two flags, one for each team
            self.objectives.append(
                Objective(
                    type=ObjectiveType.CAPTURE_FLAG,
                    position=Position(200, 200),
                    team=0,
                    description="Capture the Red Flag"
                )
            )
            self.objectives.append(
                Objective(
                    type=ObjectiveType.CAPTURE_FLAG,
                    position=Position(GAME_WIDTH - 200, GAME_HEIGHT - 200),
                    team=1,
                    description="Capture the Blue Flag"
                )
            )
        elif self.settings.game_mode == "defend_the_base":
            # Each team defends their base
            for team in range(MAX_TEAMS):
                base_pos = self.settings.spawn_points[team]
                self.objectives.append(
                    Objective(
                        type=ObjectiveType.DEFEND_BASE,
                        position=base_pos,
                        team=team,
                        radius=150,
                        description=f"Defend Team {team+1}'s Base"
                    )
                )
        elif self.settings.game_mode == "escort_payload":
            # One payload to escort to the enemy base
            payload_pos = Position(GAME_WIDTH // 2, GAME_HEIGHT // 2)
            self.objectives.append(
                Objective(
                    type=ObjectiveType.ESCORT_PAYLOAD,
                    position=payload_pos,
                    team=-1,
                    max_progress=150.0,
                    radius=80,
                    description="Escort the Payload to the Enemy Base"
                )
            )
        else:  # team_deathmatch - no objectives
            pass
    
    def spawn_power_ups(self, count: int) -> None:
        """Spawn power-ups at random locations"""
        for _ in range(count):
            # Avoid spawn points and objectives
            while True:
                x = random.uniform(100, GAME_WIDTH - 100)
                y = random.uniform(100, GAME_HEIGHT - 100)
                pos = Position(x, y)
                
                # Check distance from spawn points and objectives
                too_close = False
                for spawn in self.settings.spawn_points:
                    if pos.distance_to(spawn) < 150:
                        too_close = True
                        break
                
                if too_close:
                    continue
                
                for obj in self.objectives:
                    if pos.distance_to(obj.position) < 100:
                        too_close = True
                        break
                
                if not too_close:
                    break
            
            # Random power-up type
            power_up_types = list(PowerUpType)
            power_up_type = random.choice(power_up_types)
            
            power_up = PowerUp(
                position=pos,
                type=power_up_type,
                spawn_time=time.time()
            )
            self.power_ups.append(power_up)
    
    def update(self, delta_time: float) -> None:
        """Update game state"""
        if not self.game_running:
            return
            
        self.current_time = time.time()
        elapsed = self.current_time - self.last_update_time
        
        # Update game stats
        self.stats.update(elapsed)
        
        # Update time-based systems
        self.update_environment(elapsed)
        self.update_robot_ai(elapsed)
        self.update_robot_movement(elapsed)
        self.update_projectiles(elapsed)
        self.update_power_ups(elapsed)
        self.update_objectives(elapsed)
        self.update_team_scores(elapsed)
        
        # Check game end condition
        self.check_game_end()
        
        self.last_update_time = self.current_time
    
    def update_environment(self