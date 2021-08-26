import os, sys, random, operator
import numpy as np

etemax = 0.37
etemin = 0.04

class Environment:
    
    
    
    def __init__(self, Ny=2, Nx=2):
        # Definir el espacio de estados
        self.Ny = Ny  # tamaño cuadrícula Y
        self.Nx = Nx  # tamaño cuadrícula X
        self.state_dim = (Ny, Nx)
        
        # Definir espacio de acción
        self.action_dim = (4,)  # up, right, down, left
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # traducciones de los estados
       
        # Definir tabla de recompensas
        self.R = self._build_rewards()  # R(s,a) recompensas del agente
        
        # Verificar la coherencia del espacio de acción
        if len(self.action_dict.keys()) != len(self.action_coords):
            exit("err: inconsistent actions given")

    def reset(self):
        # Restablecer el estado del agente en la esquina superior izquierda de la cuadrícula
        self.state = (0, 0)  
        return self.state

    def step(self, action):
        # Evolucionar el estado del agente
        state_next = (self.state[0] + self.action_coords[action][0],
                      self.state[1] + self.action_coords[action][1])
      
        # Recoger la recompensa
        reward = (1/(etemin/etemax))
        
        # Terminar si llegamos a la esquina inferior derecha de la cuadrícula
        done = (state_next[0] == self.Ny - 1) and (state_next[1] == self.Nx - 1)
        
        # Estado de actualización
        self.state = state_next
        return state_next, reward, done
    
    def allowed_actions(self):
        # Genere una lista de acciones permitidas según la ubicación de la red del agente
        actions_allowed = []
        y, x = self.state[0], self.state[1]
        if (y > 0):  # no pasar el límite superior
            actions_allowed.append(self.action_dict["up"])
        if (y < self.Ny - 1):  # sin límite inferior de paso
            actions_allowed.append(self.action_dict["down"])
        if (x > 0):  # sin límite de paso a la izquierda
            actions_allowed.append(self.action_dict["left"])
        if (x < self.Nx - 1):  # sin límite derecho de paso
            actions_allowed.append(self.action_dict["right"])
        actions_allowed = np.array(actions_allowed, dtype=int)
        return actions_allowed

    def _build_rewards(self):
        # Definir recompensas para agentes R[s,a]
        r_goal = (1/(etemin/etemax))  # recompensa por llegar al estado terminal (esquina inferior derecha)
        r_nongoal = (etemin/etemax)  # penalización por no llegar al estado terminal
        R = r_nongoal * np.ones(self.state_dim + self.action_dim, dtype=float)  # R[s,a]
        R[self.Ny - 2, self.Nx - 1, self.action_dict["down"]] = r_goal  # llegar desde arriba
        R[self.Ny - 1, self.Nx - 2, self.action_dict["right"]] = r_goal  # llegar por la izquierda
        return R

class Agent:
    
    def __init__(self, env):
        # Estado y dimensión de acción 
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        # Parámetros de aprendizaje del agente
        self.epsilon = 1.0  # probabilidad de exploración inicial
        self.epsilon_decay = 0.99  # decaimiento de épsilon después de cada episodio
        self.beta = 0.99  # tasa de aprendizaje
        self.gamma = 0.99  # factor de descuento de recompensa
        # Inicializar tabla Q[s,a]
        self.Q = np.zeros(self.state_dim + self.action_dim, dtype=float)

    def get_action(self, env):
        # Política de agente codicioso de épsilon
        if random.uniform(0, 1) < self.epsilon:
            # explorar
            return np.random.choice(env.allowed_actions())
        else:
            # explotar las acciones permitidas
            state = env.state;
            actions_allowed = env.allowed_actions()
            Q_s = self.Q[state[0], state[1], actions_allowed]
            actions_greedy = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            return np.random.choice(actions_greedy)

    def train(self, memory):
        (state, action, state_next, reward, done) = memory
        sa = state + (action,)
        self.Q[sa] += self.beta * (reward + self.gamma*np.max(self.Q[state_next]) - self.Q[sa])


    def display_greedy_policy(self):
        greedy_policy = np.zeros((self.state_dim[0], self.state_dim[1]), dtype=int)
        for x in range(self.state_dim[0]):
            for y in range(self.state_dim[1]):
                greedy_policy[y, x] = np.argmax(self.Q[y, x, :])
        print("\nGreedy policy(y, x):")
        print(greedy_policy)
        print()

# Ajustes
env = Environment(Ny=2, Nx=2)
agent = Agent(env)

# Entrenamiento del agente
print("\nTraining agent...\n")
N_episodes = 1 
for episode in range(N_episodes):

    # Genera un episodio
    iter_episode, reward_episode = 0, 0
    state = env.reset()  # estado inicial
    while True:
        action = agent.get_action(env)  # conseguir acción
        state_next, reward, done = env.step(action)  # evolucionar estado por acción
        agent.train((state, action, state_next, reward, done))  # Entrenar agente
        iter_episode += 1
        reward_episode += reward
        if done:
            break
        state = state_next  # transición al siguiente estado

    # Parámetro de exploración de decaimiento del agente
    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, 0.2)

    # Imprimir
    if (episode == 0) or (episode + 1) % 2 == 0:
        print("\n [episode {}/{}] iter = {}, rew = {:.1F}".format(
            episode + 1, N_episodes, iter_episode, reward_episode))
        if((iter_episode == 10) or (iter_episode == 9) or (iter_episode == 8)):
            print("\n nodo estable")
        else: 
             print("\n nodo inestable")  

