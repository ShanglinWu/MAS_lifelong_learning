from setuptools import find_packages, setup


setup(
    name="llma-mem",
    version="0.1.0",
    description="Focused LLMA-Mem package for lifelong multi-agent memory experiments.",
    packages=find_packages(include=["llmamem", "llmamem.*"]),
    include_package_data=True,
    python_requires=">=3.10",
)
