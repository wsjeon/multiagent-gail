from setuptools import setup, find_packages

setup(name='multiagent-gail',
      version='0.0.1',
      description='Multi-Agent Generative Adversarial Imitation Learning',
      url='https://github.com/ermongroup/multiagent-gail',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl']
)
