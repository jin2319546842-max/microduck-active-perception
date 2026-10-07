import unittest
import numpy as np
from training.env import NavigationEnv, sample_scene

class EnvironmentTests(unittest.TestCase):
    def test_reproducible_split_and_mirror(self):
        self.assertEqual(sample_scene(2,8),sample_scene(2,8))
        self.assertNotEqual(sample_scene(2,8),sample_scene(2,8,'test'))
        self.assertEqual({sample_scene(1,i)['mirror'] for i in range(20)},{-1,1})

    def test_observation_and_targets_are_separate(self):
        env=NavigationEnv(1);h=env.reset(4)
        self.assertEqual(h.shape,(8,135));self.assertEqual(env.target().shape,(12,))
        self.assertTrue(np.isfinite(h).all())
        # Labels can be changed without changing the actor observation.
        a=NavigationEnv(0);b=NavigationEnv(0);a.reset(1);b.reset(1)
        b.target=lambda:np.zeros(12)
        np.testing.assert_array_equal(a.observation(),b.observation())

    def test_head_actuation_changes_tof(self):
        env=NavigationEnv(1);before=env.reset(2)[-1,:64]
        for _ in range(6):h,*_=env.step([0,0,1])
        self.assertGreater(env.head,.8)
        self.assertGreater(np.max(np.abs(before-h[-1,:64])),.1)

    def test_collision_and_timeout(self):
        env=NavigationEnv(1);env.reset(0);env.pos=np.array([0.,-.2]);env.yaw=np.pi/2
        env.walls=np.array([[0.,0.,4.,.1]])
        for _ in range(10):
            _,_,done,truncated,info=env.step([1,0,0])
            if done:break
        self.assertTrue(info['collision']);self.assertTrue(done);self.assertFalse(truncated)
        env=NavigationEnv(0,max_steps=1);env.reset(0)
        _,_,done,truncated,_=env.step([0,0,0])
        self.assertFalse(done);self.assertTrue(truncated)
        with self.assertRaises(RuntimeError):env.step([0,0,0])

    def test_reset_collision_free_and_goal_reachable(self):
        # Discretized flood fill checks representative randomized scene geometry.
        for stage in range(5):
            for seed in range(8):
                env=NavigationEnv(stage);env.reset(seed)
                self.assertGreater(env.clearance(env.pos),.12)
                self.assertGreater(env.clearance(env.goal),.12)
                grid=np.arange(-3.3,3.31,.15)
                start=tuple(np.argmin(abs(grid-v)) for v in env.pos)
                goal=tuple(np.argmin(abs(grid-v)) for v in env.goal)
                stack=[start];seen={start}
                while stack and goal not in seen:
                    i,j=stack.pop()
                    for x,y in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
                        if 0<=x<len(grid) and 0<=y<len(grid) and (x,y) not in seen and env.clearance([grid[x],grid[y]])>.12:
                            seen.add((x,y));stack.append((x,y))
                self.assertIn(goal,seen,(stage,seed))

if __name__=='__main__':unittest.main()
