from ipykernel.kernelapp import IPKernelApp
from . import SilikBaseKernel

IPKernelApp.launch_instance(kernel_class=SilikBaseKernel)
