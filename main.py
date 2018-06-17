import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from Market import Market
from Agent import DeepQNetwork1


def run():
    rw=[]
    step = 0
    observation = env.ot[3:]
    print(len(observation))
    print(observation)
    for episode in range(2000):
#     while True:
        
        # RL choose action based on observation
        action = RL.choose_action(observation)

        # RL take action and get next observation and reward
        observation_, reward = env.step(action)
        rw=rw+[reward]
        
        RL.store_transition(observation, action, reward, observation_)

        if (step > 200) and (step % 5 == 0):
            RL.learn()

        # swap observation
        observation = observation_

        step += 1
    print("交易次數：",sum(env.action_record.action!=0))
    print("手續費：",sum(env.action_record.action!=0)*0.5*50)
    print("賺賠點數：",reward)
    print("累計損益：",reward*50)
    print("累計損益+手續費：",reward*50-sum(env.action_record.action!=0)*0.5*50)
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    plt.plot(env.action_record[["price"]])
    plt.plot(rw)
    plt.savefig("price_pl.png")
    plt.close()
    plt.plot(env.action_record[["action"]].cumsum())
    plt.savefig("action.png")
    plt.close()
    
    RL.plot_cost()


if __name__ == "__main__":
    # maze game
    env = Market()
    RL = DeepQNetwork1(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True,
                      max_position=5
                      )
    run()

 