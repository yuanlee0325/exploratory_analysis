from opentrons.simulate import simulate, format_runlog
import argparse
from pathlib import Path

path = Path('protocol.py')

def run_simulation(path):
    try:
        with open(path) as protocol_file:
            print('Successfully loaded protocol!')
            runlog, _bundle = simulate(protocol_file) 
            print(format_runlog(runlog))
    except Exception as e:
        print(f"❌ Simulation failed: {e}")

if __name__ == '__main__':  
    parser = argparse.ArgumentParser(
        description='Simulate an opentrons experiment.'
    )
    parser.add_argument('-i', help='path')
    args = parser.parse_args()
    run_simulation(args.i)

