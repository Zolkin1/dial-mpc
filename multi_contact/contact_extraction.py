import os
import time
import importlib
import sys
import numpy as np
import math
from dataclasses import dataclass, replace

import yaml
import argparse

import art
import emoji
from numpy.core.records import record

import dial_mpc.envs as dial_envs
from dial_mpc.config.base_env_config import BaseEnvConfig
from dial_mpc.core.dial_core import DialConfig, MBDPI
from dial_mpc.utils.io_utils import (
    load_dataclass_from_dict,
    get_model_path,
    get_example_path,
)

import jax
from jax import numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

import brax.envs as brax_envs
from brax.io import html

import mujoco
import mujoco.viewer

@dataclass
class ContactExtractorConfig:
    robot_name: str
    scene_name: str
    sim_leg_control: str
    visualize: bool
    sim_dt: float       # TODO: Keep?
    contact_bodies: [str]

class ContactExtractor:
    def __init__(
            self,
            sim_config: ContactExtractorConfig,
    ):
        self.visualize = sim_config.visualize
        self.mj_model = mujoco.MjModel.from_xml_path(
            get_model_path(sim_config.robot_name, sim_config.scene_name).as_posix()
        )
        # NOTE: This must match the ctrl_dt for dial mpc
        self.mj_model.opt.timestep = sim_config.sim_dt
        self.mj_data = mujoco.MjData(self.mj_model)

        self.contact_bodies = sim_config.contact_bodies

        # get home keyframe
        self.default_q = self.mj_model.keyframe("home").qpos
        self.default_u = self.mj_model.keyframe("home").ctrl

       # mujoco setup
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.viewer = mujoco.viewer.launch_passive(
            self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
        )

    def record_contacts(self, step_num):
        for i in range(self.mj_data.ncon):
            geom1_id = self.mj_data.contact[i].geom[0]
            geom2_id = self.mj_data.contact[i].geom[1]

            body1_id = self.mj_model.geom_bodyid[geom1_id]
            body2_id = self.mj_model.geom_bodyid[geom2_id]

            body1_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

            # TODO: Determine surface and polytope representation
            # Option 1:
            #   Confirm that the other geom is a box
            #   Look at the size, position, and orientation params which will give me all the info to re-create all the surfaces
            #   Determine the surface closest to the contact point
            #   Convert that surface into a polytope representation, also somehow give the plane on which the polytope lies
            # Option 2:
            #   Look at the mesh associated with the other geom (I need to confirm that this works for the box)
            # For now, go with option 1

            if ((body1_name in self.contact_bodies and body2_name not in self.contact_bodies)
                    or (body2_name in self.contact_bodies and body1_name not in self.contact_bodies)):
                # print("body1 name: ", body1_name, " body2 name: ", body2_name, " step num: ", step_num)
                if body1_name in self.contact_bodies:
                    self.contacts[self.contact_bodies.index(body1_name)][step_num] = body2_name
                else:
                    self.contacts[self.contact_bodies.index(body2_name)][step_num] = body1_name

    def extract_contacts(self, inputs, initial_state, task, t0: float, time_horizon: float):
        num_steps = math.ceil(time_horizon/self.mj_model.opt.timestep)
        # print("num_steps: ", num_steps)
        self.contacts = [["" for _ in range(num_steps)] for _ in range(len(self.contact_bodies))]
        print(self.contacts)
        contact_times = np.linspace(t0, time_horizon + t0, num_steps)
        self.mj_data.time = t0
        step_num = 0

        # Set initial condition
        self.mj_data.qpos = initial_state.qpos
        self.mj_data.qvel = initial_state.qvel

        state = initial_state

        self.viewer.sync()

        while self.mj_data.time + self.mj_model.opt.timestep < time_horizon + t0:
            # Apply the input
            u = task.act2tau(inputs[step_num], state)
            self.mj_data.ctrl = u

            # Step the simulator
            mujoco.mj_step(self.mj_model, self.mj_data)
            state = replace(state, qpos=self.mj_data.qpos, qvel=self.mj_data.qvel)

            self.viewer.sync()
            time.sleep(0.02)
            # input("Press Enter to continue...")

            # Update other info
            step_num += 1
            self.record_contacts(step_num)

        return self.contacts, contact_times, self.contact_bodies

def main():

    def reverse_scan(rng_Y0_state, factor):
        rng, Y0, state = rng_Y0_state
        rng, Y0, info = mbdpi.reverse_once(state, rng, Y0, factor)
        return (rng, Y0, state), info

    art.tprint("Contact Extraction", font="medium", chr_ignore=True)
    parser = argparse.ArgumentParser()
    config_or_example = parser.add_mutually_exclusive_group(required=True)
    config_or_example.add_argument("--config", type=str, default=None)
    config_or_example.add_argument("--example", type=str, default=None)
    config_or_example.add_argument("--list-examples", action="store_true")
    parser.add_argument(
        "--custom-env",
        type=str,
        default=None,
        help="Custom environment to import dynamically",
    )
    args = parser.parse_args()

    if args.custom_env is not None:
        sys.path.append(os.getcwd())
        importlib.import_module(args.custom_env)

    if args.example is not None:
        config_dict = yaml.safe_load(open(get_example_path(args.example + ".yaml")))
    else:
        config_dict = yaml.safe_load(open(args.config))

    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    rng = jax.random.PRNGKey(seed=dial_config.seed)

    # find env config
    env_config_type = dial_envs.get_config(dial_config.env_name)
    env_config = load_dataclass_from_dict(
        env_config_type, config_dict, convert_list_to_array=True
    )

    print(emoji.emojize(":rocket:") + "Creating environment")
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)
    mbdpi = MBDPI(dial_config, env)

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env(rng_reset)

    YN = jnp.zeros([dial_config.Hnode + 1, mbdpi.nu])

    rng_exp, rng = jax.random.split(rng)
    # Y0 = mbdpi.reverse(state_init, YN, rng_exp)
    Y0 = YN

    contact_config =  load_dataclass_from_dict(ContactExtractorConfig, config_dict)
    contact_extractor = ContactExtractor(contact_config)

    # TODO: Put back
    Nstep = 100 #2 #dial_config.n_steps
    rews = []
    rews_plan = []
    rollout = []
    state = state_init
    us = []
    infos = []
    for t in range(Nstep):
        # forward single step
        state = step_env(state, Y0[0])
        rollout.append(state.pipeline_state)
        rews.append(state.reward)
        us.append(Y0[0])

        # update Y0
        Y0 = mbdpi.shift(Y0)

        n_diffuse = dial_config.Ndiffuse
        if t == 0:
            n_diffuse = dial_config.Ndiffuse_init
            print("Performing JIT on DIAL-MPC")

        t0 = time.time()
        traj_diffuse_factors = (
            dial_config.traj_diffuse_factor ** (jnp.arange(n_diffuse))[:, None]
        )
        (rng, Y0, _), info = jax.lax.scan(
            reverse_scan, (rng, Y0, state), traj_diffuse_factors
        )
        rews_plan.append(info["rews"][-1].mean())
        infos.append(info)
        freq = 1 / (time.time() - t0)

        print("Time horizon: ", mbdpi.step_us[-1])
        print("Y0: ", Y0)
        print("Y0 Shape: ", Y0.shape)
        print("inputs from Y0: ", mbdpi.node2u(Y0[:, 1]))
        u = np.zeros((len(mbdpi.step_us), len(Y0[1])))
        print("u: ", u)
        for i in range(len(Y0[1])):
            u[:, i] = mbdpi.node2u(Y0[:, i])
        print("u: ", u)

        print("shape: ", u.shape) #mbdpi.node2u(Y0[:, 1]).shape)
        # print("Pipeline state: ", state.pipeline_state)
        contacts, contact_times, contact_bodies = contact_extractor.extract_contacts(u, state.pipeline_state, env, 0, mbdpi.step_us[-1])
        print("contacts: ", contacts)
        print("contact_times: ", contact_times)
        print("contact_bodies: ", contact_bodies)

        # pbar.set_postfix({"rew": f"{state.reward:.2e}", "freq": f"{freq:.2f}"})

    rew = jnp.array(rews).mean()
    print(f"mean reward = {rew:.2e}")

    # save us
    # us = jnp.array(us)
    # jnp.save("./results/us.npy", us)

    # create result dir if not exist
    if not os.path.exists(dial_config.output_dir):
        os.makedirs(dial_config.output_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # plot rews_plan
    # plt.plot(rews_plan)
    # plt.savefig(os.path.join(dial_config.output_dir,
    #             f"{timestamp}_rews_plan.pdf"))

    # host webpage with flask
    print("Processing rollout for visualization")
    import flask

    app = flask.Flask(__name__)
    webpage = html.render(
        env.sys.tree_replace({"opt.timestep": env.dt}), rollout, 1080, True
    )

    # save the html file
    with open(
        os.path.join(dial_config.output_dir, f"{timestamp}_brax_visualization.html"),
        "w",
    ) as f:
        f.write(webpage)

    # save the rollout
    data = []
    xdata = []
    for i in range(len(rollout)):
        pipeline_state = rollout[i]
        data.append(
            jnp.concatenate(
                [
                    jnp.array([i]),
                    pipeline_state.qpos,
                    pipeline_state.qvel,
                    pipeline_state.ctrl,
                ]
            )
        )
        xdata.append(infos[i]["xbar"][-1])
    data = jnp.array(data)
    xdata = jnp.array(xdata)
    jnp.save(os.path.join(dial_config.output_dir, f"{timestamp}_states"), data)
    jnp.save(os.path.join(dial_config.output_dir, f"{timestamp}_predictions"), xdata)

    @app.route("/")
    def index():
        return webpage

    app.run(port=5000)


if __name__ == "__main__":
    main()