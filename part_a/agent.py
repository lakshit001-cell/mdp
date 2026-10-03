from collections import deque,defaultdict
import time
class Agent:

    def __init__(self, layout_file, prob_file):
        self.grid={}
        self.actions= [(1, 0), (-1, 0), (0, -1), (0, 1)]

        with open(layout_file, 'r') as f:
            lines = f.read().splitlines()

        self.N=len(lines)
        treasures=[]
        pirate1=None
        pirate2=None

        for li in range (self.N):
            row=lines[li]
            i = li
            for j in range(len(row)):
                char=row[j]
                loc=(i,j)

                self.grid[loc]=char
                if char=="1":
                    pirate1=loc
                elif char=="2":
                    pirate2=loc
                elif char=="S":
                    ship_start=loc
                elif char=="T":
                    treasures.append(loc)
                elif char=="F":
                    self.fort=loc

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
        ship = tuple(ship_location)
        tr = tuple(sorted([tuple(t) for t in treasure_locations]))
        
        # Pad the incoming pirates list so it always has exactly 2 elements
        p_locs = [tuple(p) if p else None for p in pirate_locations]
        while len(p_locs) < 2:
            p_locs.append(None)
            
        # Check all possible ways env.py might be handing us the pirates
        possible_states = [
            (ship, p_locs[0], p_locs[1], tr),  # Direct order
            (ship, p_locs[1], p_locs[0], tr),  # Swapped pirates
            (ship, None, p_locs[0], tr),       # Only P2 alive, passed at index 0
            (ship, p_locs[0], None, tr)        # Only P1 alive, passed at index 1
        ]
        
        for s in possible_states:
            if s in self.policy:
                return self.policy[s]

        # If it still fails, the state was never reached during Value Iteration
        print(f"STATE MISMATCH! Ship: {ship_location}, Pirates: {pirate_locations}")
        return 0

    def learn_policy(self, time_limit):
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
        self.time_limit=time_limit
        
        print("--- PHASE 1: State Discovery ---")
        self.value_iteration_setup()
        print(f"--- PHASE 2: Value Iteration (Total States: {len(self.states)}) ---")
        self.value_iteration()
        print("--- PHASE 3: Policy Extraction ---")
        self.find_policy()

    def value_iteration_setup(self):

        self.state_list= []
        self.T_cache = defaultdict(list)
        self.states =set([self.start_state])
        q =deque([self.start_state])
        states_processed=0
        while q:
            if time.time() - self.start_time >= self.time_limit - 5.0:
                print(f"Setup timeout! Processed {states_processed} states.")
                break
            state=q.popleft()
            self.state_list.append(state)
            states_processed+=1
            if states_processed % 1000 == 0:
                print(f"Discovered {states_processed} states...")
            for action in range(0,4):
                transitions=self.compute_transitions(state,action)
                self.T_cache[(state,action)]=transitions

                for next_state, prob, reward, is_term in transitions:
                    if next_state not in self.states:
                        self.states.add(next_state)
                        if not is_term:
                            q.append(next_state)


    def value_iteration(self) :
        epsilon=1e-3

        while True:
            max_change=0.0
        
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
                if change>max_change:
                     max_change=change
                self.V[state] = best_action

            if max_change<epsilon:
                 break
                

        

    def find_policy(self):
         for state in self.state_list:
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



    def state_changes(self,location, id, action):
        if location is None:
            return[(None,1)] 

        outcomes = defaultdict(float)

        for act in range(0,4):
            dx,dy=self.actions[act]
            if id==0:
                if act==action:
                    prob=self.ps
                else:
                    prob= (1.0-self.ps)/3.0
            else:
                  prob=self.p_probs[id][act]

            if prob==0.0:
                  continue

            new_x=location[0]+dx
            new_y=location[1]+dy
            is_valid=True
            if (new_x,new_y) not in self.grid:
                  is_valid=False
            else:
                char=self.grid[(new_x,new_y)]
                if char=='L':
                      is_valid=False
                elif id==1 and char not in ('1','!'):
                    is_valid=False
                elif id==2 and char not in ('2','!'):
                                    is_valid=False
            if is_valid==True:
                  outcomes[(new_x,new_y)]+= prob
            else:
                  outcomes[location]+=prob
        result=[]
        for i,j in outcomes.items():
            result.append((i,j))
        return result

          
    def compute_transitions(self,state,action):
        ship,p1,p2,treasure=state

        ship_states=self.state_changes(ship,0,action)
        p1_states=self.state_changes(p1,1,action)
        p2_states=self.state_changes(p2,2,action)

        trans_dict = {}

        for ship_location, prob_s in ship_states:
              for p1_loc, prob_p1 in p1_states:
                    for p2_loc,prob_p2 in p2_states:

                        prob=prob_s*prob_p1*prob_p2
                        if prob==0.0:
                                continue
                        reward=self.Rs
                        term=False

                        if ship_location==p1_loc or ship_location==p2_loc:
                              term=True
                              reward+=self.Rp
                        elif ship_location==self.fort:
                              term=True
                              reward+=self.Rf

                        tr=list(treasure)
                        if term==False:
                              
                                    
                            if ship_location in tr:
                                                reward+=self.Rt
                                                tr.remove(ship_location)
                        
                        next_state=(ship_location,p1_loc,p2_loc,tuple(tr))

                        if next_state not in trans_dict:
                            trans_dict[next_state] = [0.0, reward, term]
                        trans_dict[next_state][0] += prob
        return [(nxt, vals[0], vals[1], vals[2]) for nxt, vals in trans_dict.items()]




                          
            
        