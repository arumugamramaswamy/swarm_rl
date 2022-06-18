import time

def test(
    env,
    model,
    render=False,
    num_episodes=200,
):

    all_rewards = []
    for _ in range(num_episodes):

        total_reward = 0
        obs = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _ = env.step(action)

            if render:
                env.par_env.unwrapped.render()
                time.sleep(0.1)

            total_reward += rewards.sum()
            if dones.all():
                break

        all_rewards.append(total_reward)

    return all_rewards

