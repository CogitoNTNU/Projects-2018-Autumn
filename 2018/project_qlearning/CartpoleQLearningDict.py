import gym
from time import sleep, time
import keyboard
import math
import random
import matplotlib.pyplot as plt

class QLearning:
    def __init__ (self):
        self.exploration=0.6
        self.learningrate=0.3
        self.discountrate=0.99
        self.degrees=[2*math.pi*x/360 for x in range(360)]
        self.stateMatrix=dict()

    def setExploration(self, rate):
        self.exploration=rate

    def setLearningrate(self, rate):
        self.learningrate=rate

    #Choosing a random action or the best action depending on randNum
    def chooseAction(self, state):
        randNum=random.randint(0,100)
        self.stateMatrix[state] = self.stateMatrix.get(state, [0, 0])

        if randNum<=100*self.exploration:
            return random.randint(0,1)

        return 0 if self.stateMatrix[state][0]>self.stateMatrix[state][1] else 1

    #Updating the q-value
    def updateQValue(self, prevState, prevAction, newState, reward, done):
        self.stateMatrix[prevState]=self.stateMatrix.get(prevState, [0,0])
        self.stateMatrix[newState]=self.stateMatrix.get(newState, [0,0])
        if not done:
            learnedValue=self.learningrate*(reward+self.discountrate
                                   *max(self.stateMatrix[newState][0],
                                        self.stateMatrix[newState][1])
                                   -self.stateMatrix[prevState][prevAction])
        else:
            learnedValue = self.learningrate * reward
        self.stateMatrix[prevState][prevAction]+=learnedValue


    #Returning the current state given the input parameters
    def getState(self, pos, velocity, angle, angVel):
        #The different discretisations of the parameters
        posSplits=[-1.5, -1, -0.5, 0, 0.5, 1.5]
        velocitySplits=[-1, -0.5, 0, 0.5, 1]
        angleSplits=[-self.degrees[9], -self.degrees[6], -self.degrees[3], self.degrees[0], self.degrees[3],self.degrees[6], self.degrees[9]]
        angVelSplits=[-1, -0.7, -0.3, -0.1, 0, 0.1, 0.3, 0.7, 1]
        parameters=[posSplits, velocitySplits, angleSplits, angVelSplits]
        inpParamaters=[pos, velocity, angle, angVel]
        state="_"

        #Creating the state string
        #state=_pos|velocity|angle|angelVelocity|
        for x in range(len(parameters)):
            for y in range(len(parameters[x])):
                if inpParamaters[x]<parameters[x][y]:
                    state+=str(parameters[x][y])
                    break
            state+=str("|")
        return state


def addPoint(frames):
    framesVector.append(frames)
    posVector.append(len(framesVector))
    #Updating the graph every 500 session
    if len(posVector)%500==0 and showKeyPressed:
        plt.plot(posVector, framesVector)
        plt.show()
        plt.pause(0.0001)
    
    

#Setting up the agent and the environment
env = gym.make('CartPole-v0')
agent=QLearning()
highest=0
showKeyPressed = False #the key for rendering is pressed
passed10000=False

#Setting up the plot
framesVector=[] #y-coodrinates
posVector=[] #x-coordinates
plt.ion()
figure = plt.figure()


#Playing 1000000 games
while True:
    for episode in range(10000000):
        observation = env.reset()
        action=0
        curState=-1
        millis=time()
        #Upper limit of 1 mill. decisions in each episode
        for t in range(100000000):
            #Renderging the environment if requrements are met
            if t%100==0 and keyboard.is_pressed('a'):
                showKeyPressed=not showKeyPressed
                sleep(0.5)
            if t>10000:
                passed10000=True
            if (episode%500==0 and not passed10000) or t>10000:
               showKeyPressed=True
            if showKeyPressed:
                env.render()


            #execute the action
            observation, reward, done, info = env.step(action)

            #the episode is over if it tilts more than 12 degrees or is out of the screen
            done=abs(observation[0])>=2.4 or abs(observation[2])>12*math.pi/180

            # updating current- and previous state
            prevState=curState
            curState=agent.getState(observation[0], observation[1], observation[2], observation[3])

            #Update QValue
            if t>0:
                reward1=-1 if done else 1
                agent.updateQValue(prevState, action, curState, reward1-min(1, abs(observation[2]*5))-abs(observation[0]/2), done)
            #Changing exploration- and learningrate
            agent.setExploration(max(0,1-episode/2000))
            agent.setLearningrate(max(0.1, 0.4-episode/3000))

            #Finding next action
            action=agent.chooseAction(curState)

            #Game is finished
            if done:
                highest=max(highest, t) #longest time alive
                print("Episode finished after {} timesteps".format(t+1)+ " attempt "+str(episode)+" highest ", highest)
                #addPoint(t) #adding point to graph
                showKeyPressed=False
                break
