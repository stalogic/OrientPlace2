from gym.envs.registration import register

register(id="place_env-v0", entry_point="place_env.place_env:PlaceEnv")


register(id="orient_env-v0", entry_point="place_env.orient_env_rust:OrientPlaceEnv")
