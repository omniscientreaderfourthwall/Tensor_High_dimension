import subprocess

'''
This part will be composed of file name: 

TRAIN4_1A, 

This is for point 1.

'''
program_list = ['Example_3_32float.py', 'Example_3_64float.py', 
                'Example_3_withboundary.py', 'Example_3_withoutboundary.py',
                'Example_3_Mixed.py', 'Example_3_Non_Mixed.py']

for program in program_list:
    subprocess.call(['python', program])
    print("Finished:" + program)


from google.colab import runtime

runtime.unassign()

