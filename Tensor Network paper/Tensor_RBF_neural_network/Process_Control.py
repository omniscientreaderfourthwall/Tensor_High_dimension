import subprocess

'''
This part will be composed of file name: 

TRAIN4_1A, 

This is for point 1.

'''
program_list = ['Example_5.py']

for program in program_list:
    subprocess.call(['python', program])
    print("Finished:" + program)


from google.colab import runtime

runtime.unassign()

