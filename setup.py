import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch_custom_losses",
    version="0.0.1",
    author="Lukas Wolf",
    author_email="luwu@example.com",
    description="Pytorch custom loss functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lu-wo/pytorch-custom-losses",
    package_dir={"": "pytorch_custom_losses"},
    packages=setuptools.find_packages(where="pytorch_custom_losses"),
    python_requires=">=3.6",
)
