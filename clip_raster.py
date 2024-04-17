import glob
from cdo import Cdo
cdo = Cdo()
cdo.debug = True

input_location = 'PimaCo' # location in filename of input files
output_location = 'EPC'  # updated location of clipped output

for f in glob.glob(f'{input_location}_*.nc'):
    print(f'Clipping {f}')
    out_f = f.replace(f'{input_location}', f'{output_location}')
    cdo.sellonlatbox(-111.41,-110.43,31.7,32.54, input=f, output=out_f, options='-z zip')

print('done')
