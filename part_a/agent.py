from collections import deque,defaultdict
import time
import heapq
class Agent:

    def __init__(self, layout_file, prob_file):

        self.grid={}
        self.actions = [(0, 1), (0, -1), (-1, 0), (1, 0)]

        with open(layout_file, 'r') as f:
            lines = f.read().splitlines()

        self.N=len(lines)
        treasures=[]

        for i in range (self.N):
            row=lines[i]
            for j in range(len(row)):
                char=row[j]
                grid_i= self.N-1-i
                self.grid[(j,grid_i)]=char
                if char=="1":
                    pirate1=(j,grid_i)
                elif char=="2":
                    pirate2=(j,grid_i)
                elif char=="S":
                    ship_start=(j,grid_i)
                elif char=="T":
                    treasures.append=(j,grid_i)
                elif char=="F":
                    self.fort=(j,grid_i)

        treasures.sort()
        self.start_state= (ship_start,pirate1,pirate2,tuple(treasures))

        with open(prob_file, 'r') as f:
            lines = f.read().splitlines()

        
        self.ps=float(lines[0])
        self.p1pr = []
        for p in lines[1].split():
            self.p1pr.append(float(p))
        self.p2pr = []
        for p in lines[2].split():
            self.p2pr.append(float(p))


        self.p_probs = {1: self.p1pr, 2: self.p2pr}
        rewards=lines[3].split()
        self.Rs=float(rewards[0])
        self.Rt=float(rewards[1])
        self.Rf=float(rewards[2])
        self.Rp=float(rewards[3])
        self.gamma=float(lines[4])

        self.policy={}
        self.V=defaultdict(float)

    def get_action(self, ship_location, pirate_locations, treasure_locations) -> int:
        """
        Choose an action for the current state.

        Args:
            ship_location: Tuple (x, y) representing the current
                location of the ship.

            pirate_locations: List containing the locations of all
                pirates. There can be at most 2 pirates.

            treasure_locations: List containing the locations of all
                treasures. There can be at most 2 treasures.

        Returns:
            An integer representing the selected action:

                0 -> UP
                1 -> DOWN
                2 -> LEFT
                3 -> RIGHT

        Note:
            This function may be called multiple times after
            `learn_policy()` has been executed.
        """
        p1=tuple(pirate_locations[0]) if len(pirate_locations)>0 else None
        p2=tuple(pirate_locations[1]) if len(pirate_locations)>1 else None
        state=(ship_location,p1,p2,tuple(sorted(treasure_locations)))

        if state in self.policy:
            return(self.policy[state])
        else:
            return 0

    def learn_policy(self, time):
        """
        Learn a policy for navigating the Treasure Hunt environment.

        Args:
            time: Maximum time (in seconds) allowed for learning.

        The learned policy should be stored internally by the agent
        and subsequently used by `get_action()`.

        Returns:
            None
        """

        self.start_time=time.time()
        self.time_limit=self.time_limit
        
        
        self.value_iteration_setup()
        self.value_iteration()
        self.find_policy()

    def value_iteration_setup(self):
        self.predecessors =defaultdict(set)
        self.T_cache = defaultdict(list)
        self.states =set([self.start_state])
        q =deque([self.initial_state])

        while q:
            state=q.popleft()
            for action in range(0,4):
                transitions=self._compute_transition(state,action)
                self.T_cache[(state,action)]=transitions

                for next_state, prob, reward, is_term in transitions:
                    self.predecessors[next_state].add(state)
                    if next_state not in self.states:
                        self.states.add(next_state)
                        if not is_term:
                            q.append(next_state)


    def value_iteration(self) :
        epsilon=1e-3
        pq=[]
        for state in self.states:
            best_action=float('-inf')
            for action in range(0,4):
                q_value=0
                for next_state, prob, reward, term in self.T_cache[(state,action)]:
                    if term:
                        #valuation of terminal state is zero
                        q_value=q_value+prob*reward
                    else:
                        q_value= q_value+ prob*(reward+ self.gamma*self.V[next_state])
                if q_value> best_action:
                    best_action=q_value

            change= abs(best_action-self.V[state])
            if change>epsilon:
                heapq.heappush(pq,(-change,state))

        while len(pq)!=0:
            if time.time() - self.start_time >= self.time_limit - 1.5:
                break

            change, state=heapq.heappop(pq)

            best_action=float('-inf')
            for action in range(0,4):
                q_value=0
                for next_state, prob, reward, term in self.T_cache[(state,action)]:
                                if term:
                                    #valuation of terminal state is zero
                                    q_value=q_value+prob*reward
                                else:
                                    q_value= q_value+ prob*(reward+ self.gamma*self.V[next_state])
                if q_value> best_action:
                                best_action=q_value

            self.V[state]=best_action

            for previous_state in self.predecessors[state]:
                p_best_action=float('-inf')
                for action in range(0,4):
                                 q_value=0
                                 for next_state, prob, reward, term in self.T_cache[(state,action)]:
                                                 if term:
                                                     #valuation of terminal state is zero
                                                     q_value=q_value+prob*reward
                                                 else:
                                                     q_value= q_value+ prob*(reward+ self.gamma*self.V[next_state])
                                 if q_value> p_best_action:
                                                 p_best_action=q_value

                p_change=abs(p_best_action-self.V[previous_state])
                if p_change>epsilon:
                     heapq.heappush(pq,(-p_change,previous_state))


    def find_policy(self):
         for state in self.states:
            pol_act=0
            best_action=float('-inf')
            for action in range(0,4):
                              q_value=0
                              for next_state, prob, reward, term in self.T_cache[(state,action)]:
                                              if term:
                                                  #valuation of terminal state is zero
                                                  q_value=q_value+prob*reward
                                              else:
                                                  q_value= q_value+ prob*(reward+ self.gamma*self.V[next_state])
                              if q_value> best_action:
                                              best_action=q_value
                                              pol_act=action
            self.policy[state]=pol_act


    def compute_transitions(self,state,action):
                
         