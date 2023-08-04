import os
import pandas as pd
import numpy as np
import scipy.io


if __name__ == "__main__":
    
    maskmat = 'out.mat'
    csvfolder = 'tracking'
    measfolder = 'roi_measurements'
    outfolder = 'sortedByCell'
    
    

def classify_and_write_csv(maskmat,csvfolder,measfolder,outfolder,
                                    maskvarname = 'roimasks',
                                    verbose=True):
    """
classify_and_write_csv(maskmat,csvfolder,measfolder,outfolder,
                                maskvarname = 'roimasks',
                                verbose=True)
                                
classifies all of the trajectories in a csv file based on the ROI mask in 
the .mat file maskmat

inputs:
maskmat - .mat file that contains the segmented masks
csvfolder - folder that contains the tracked csvs
measfolder - folder that contains cell measurements and metadata for each FOV
outfolder - output folder for writing out classified csvs
maskvarname - name of variable within the .mat file that contains the 
relevant masks [default 'roimasks']
   (in earlier versions of cellpicker, this variable was called 'masks')

output:
returns a dataframe with cell measurements for each FOV
    """    
    # column number in which trajectory index is stored in tracking csv files
    TRAJECTORY_COLUMN = 17 
    
    os.makedirs(outfolder,exist_ok = True)
    masks = scipy.io.loadmat(maskmat)
    nrange = masks['range'][0]     # range of file numbers (upper and lower bounds inclusive)
    isfirst = True
    
    # variable specifying whether or not each FOV had selected cells or not
    FOV_is_selected = masks['classification'][0]
    
    for j in range(nrange[0],nrange[1]+1):

        # cell masks in current FOV
        currmask = masks[maskvarname][0][j-1] 
        # vector of categories in which each cell was classified by the user
        cell_categories = masks['categories'][0][j-1][0]

        # skip this FOV if the mask is empty or if the FOV is marked unselected
        if currmask.size==0 or FOV_is_selected[j-1]==0:
            continue

        # Display status in the terminal, overwriting previous output
        if verbose:
            print(f'Processing FOV {j}.', end="\r")
        
        try:
            fname = f"{csvfolder}/{j}.csv"
            
            # Loop over the file a first time to get maximum trajectory number in the file
            maxtraj = 0
            with open(fname) as fh:
                fh.readline()
                line = fh.readline()
                while line:
                    maxtraj = max(int(line.split(',')[TRAJECTORY_COLUMN]),maxtraj)
                    line = fh.readline()
                    
            # initialize an array of -1's with that many elements to contain the cell number 
            # for each trajectory (or NaN if the trajectory passes over multiple cells)
            trajcell = -np.ones(maxtraj+1) # array starting at 0 and ending at maxtraj, inclusive
            
            # loop over the csv file a second time, and determine in which cell mask each trajectory falls
            with open(fname) as fh:
                fh.readline()
                line = fh.readline()
                allcelln = set()
                while line:
                    linespl = line.split(',')
                    
                    # current trajectory number
                    trajn = int(linespl[TRAJECTORY_COLUMN])
                    
                    # current x and y coordinates
                    x = round(float(linespl[1]))
                    y = round(float(linespl[0]))
                    
                    # get cell number
                    # celln = 0 corresponds to background regions. Numbering of cells starts at 1
                    celln = currmask[y,x] 
                    
                    # add this cell index to the list of all cell indices
                    allcelln.add(celln)
                    
                    # if it has not yet been classified, classify it to the cell it is in
                    if trajcell[trajn] == -1: 
                        trajcell[trajn] = celln
                    # if it has previously been classified to another cell, set it to nan
                    elif trajcell[trajn] != celln:
                        trajcell[trajn] = np.nan
                    line = fh.readline()
            
            # loop over the file one last time and write out each line to a file for that cell
            with open(fname) as fh:
                header = fh.readline()

                # open output file handles and initialize each with a header row
                fh2 = {}
                for n in allcelln:
                    # category in which this cell was classified by the user
                    currcat = cell_categories[n-1]
                    # only generate an output file for this cell if it is selected 
                    # (i.e., if it is assigned a category not equal to zero)
                    # exclude trajectories in the background region (n = 0)
                    if currcat>0 and n>0:
                        os.makedirs(f"{outfolder}/{currcat}",exist_ok=True)
                        fh2[n] = open(f"{outfolder}/{currcat}/{j}_{n}.csv",'w')
                        fh2[n].write(header)

                line = fh.readline()
                while line:
                    linespl = line.split(',')
                    
                    # trajectory number of current localization
                    trajn = int(linespl[TRAJECTORY_COLUMN])
                    
                    # cell number of current trajectory
                    celln = trajcell[trajn]
                    
                    # only write out the current localization if it is part of a 
                    # trajectory within a cell that is selected
                    if not np.isnan(celln) and celln != 0:
                        if cell_categories[int(celln)-1] != 0:
                            celln = int(celln)
                            fh2[celln].write(line)
                    
                    line = fh.readline()
        
                # close all file handles
                for f in fh2.keys():
                    fh2[f].close()
            
            # read in measurements for all selected cells in this FOV, and append
            # to an output dataframe
            selected_cells = np.nonzero(cell_categories)[0]
            df = pd.read_csv(f"{measfolder}/{j}.csv")
            df.rename(columns={df.columns[0]: 'cellnum'}, inplace=True)
            df = df.iloc[selected_cells].copy()
            df['fovnum'] = j
            df['category'] = cell_categories[selected_cells]
            if isfirst:
                rv = df
                isfirst = False
            else:
                rv = rv.append(df,ignore_index=True)
            
        except Exception as e:
            print(f"Error with FOV {j}:", str(e))
            traceback.print_exc()
            print(f"Error occurred at line {traceback.extract_tb(e.__traceback__)[-1].lineno}: {str(e)}")
    rv.to_csv(f"{outfolder}/measurements.csv")
    return rv
    
if __name__ == "__main__":    
    classify_and_write_csv(maskmat,csvfolder,measfolder,outfolder)
    print('Finished sorting trajectories.')
    