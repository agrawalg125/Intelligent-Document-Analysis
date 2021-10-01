import os
command="pip install -r Libraries_needed.txt"
command2="python -m spacy download en_core_web_sm"
os.system('cmd /c "{}"'.format(command))

'''
uncomment when running 1st time
os.system('cmd /c "{}"'.format(command2))
'''
