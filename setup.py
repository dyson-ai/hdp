from setuptools import find_packages, setup

setup(
    name="hdp",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    license=open("LICENSE").read(),
    zip_safe=False,
    description="Hierarchical Diffusion Policy",
    author="Xiao Ma",
    author_email="xiao.ma@dyson.com",
    url="https://yusufma03.github.io/projects/hdp/",
    keywords=[
        "Diffusion Policy",
        "Behavior-Cloning",
        "Langauge",
        "Robotics",
        "Manipulation",
    ],
)
