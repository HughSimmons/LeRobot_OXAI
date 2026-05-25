from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
import time

config = SO101FollowerConfig(
    port="/dev/tty.usbmodem5AB90659861",
    id="my_awesome_follower_arm"
)

follower = SO101Follower(config)
follower.connect()

while True:
    obs = follower.get_observation()
    print(obs)
    follower.disconnect()
    time.sleep(0.1)