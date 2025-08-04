from opentrons import protocol_api

metadata = { 
            "protocolName": "test",
            "description": "test for connection",
            "author": "Yuan Li",
            "apiLevel": "2.18"
        }

# Run protocol as defined in move_commands.json
def run(protocol: protocol_api.ProtocolContext):

    """
    Execute protocol as defined in move_commands.json
    """   
    print('start home!')
    protocol.home()
    print('robot homed!')

    print('check if tips attached to the pipette')
    instruments = {}
    instruments = protocol.loaded_instruments
    if 'right' in instruments:
        print("Instrument on the right mount:", instruments['right'])
        pipette = instruments['right']
        if pipette.has_tip:
            pipette.drop_tip()
    else:
        print("No instrument on the right mount")
    if 'left' in instruments:
        print("Instrument on the left mount:", instruments['left'])
        pipette =instruments['left']
        if pipette.has_tip:
            pipette.drop_tip()
    else:
        print("No instrument on the left mount")
    print('Done!')
    del protocol

