>allocate a node e.g.
```
salloc --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=128 --time=00:59:00 --partition=cpu --account=${account_name}  --qos=dev
```
```
salloc --nodes=1 --ntasks=4 --ntasks-per-node=4 --gpus-per-task=1 --time=01:59:00 --partition=gpu --account=${account_name}  --qos=dev
```
>do

1) start with shift+enter a new python 
2) a new srun will be activated, with python running on the frontend
3) ctrl+z to stop python
4) ssh to the allocated node
5) cd in the folder where you have the nbml code

>load modules (e.g. for MeluXina cluser https://docs.lxp.lu/)
```
module load env/staging/2022.1
module load Python/3.10.4-GCCcore-11.3.0
ml SciPy-bundle/2022.05-foss-2022a 
ml PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
ml IPython/8.5.0-GCCcore-11.3.0
python
```
>Use the following to ensure correct node

shift+enter in the script
```
import sys
print(sys.executable)
import socket
hostname = socket.gethostname()
hostname
```
>Use the following to view GPUs usage. You can use the initial srun (right hand, bottom side in VSCode)
```
watch -n1 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv'
```
>put this before pytorch train epochs loop
```
with torch.backends.mkl.verbose(torch.backends.mkl.VERBOSE_OFF):
```
