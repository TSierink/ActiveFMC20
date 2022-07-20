from fmc20 import FMC20
from leaker.attack.range_database import RangeDatabase2D
import numpy as np
import random
import math

def rotate(origin, point, angle):
    """
    Method for rotating set of points anticlockwise, finding all rigid motions of a database
    The angle in radians
    """
    ox, oy = origin
    px, py = point
    
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def transform(search_points, rec, rot,s,suc, ACTUAL, match):
    """
    Method for checking all rigid motions of the database
    """
    found=False
    
    #Double check if search points are found, then check if complete match
    fit = np.all([elem in rec for elem in search_points])
    if fit: 
        if np.all([elem in rec for elem in ACTUAL]):
            print('yes')
            s += 1
            rot += 1
            suc = True 
            found = True

    if not suc: 
        # create all equivalent sets
        rec_y = np.array([a[1] for a in rec])
        rec_x = np.array([a[0] for a in rec])
        rec_yflip = np.array([-a[1] + (1 + max(rec_y)) for a in rec])
        rec_xflip = np.array([-a[0] + (1 + max(rec_x)) for a in rec])

        I_y = np.column_stack((rec_x, rec_yflip)).tolist()           #Y-FLIPPED
        I_x = np.column_stack((rec_xflip, rec_y)).tolist()           #X-FLIPPED
        I_180 = np.column_stack((rec_xflip, rec_yflip)).tolist()      #X-FLIPPED Y-FLIPPED = 180 degrees
        I_90 = [rotate(((MAX_RANGEX + 1) / 2, (MAX_RANGEY + 1) / 2), p, 0.5 * math.pi) for p in rec]
        I_270 = [rotate(((MAX_RANGEX + 1) / 2, (MAX_RANGEY + 1) / 2), p, 1.5 * math.pi) for p in rec]
        I_xy = [rotate(((MAX_RANGEX + 1) / 2,(MAX_RANGEY + 1)/2), p, 0.5 * math.pi) for p in I_y]
        I_yx = [rotate(((MAX_RANGEX +1) / 2, (MAX_RANGEY + 1) / 2), p, 0.5 * math.pi) for p in I_x]  
        
        # Cycle and search through all equivalent sets
        transformations = [I_y, I_x, I_xy, I_yx, I_90, I_180, I_270, np.flip(I_y).tolist(),np.flip(I_x).tolist(),np.flip(I_180).tolist()]
        for t in transformations:
            if not found:
                if np.all([elem in t for elem in search_points]):
                    match += 1
                    if np.all([elem in t for elem in ACTUAL]):
                        s += 1
                        rot += 1
                        suc = True 
                        found = True

    return rot, s, suc, match

# CONSTANTS
MAX_RANGEX = 30
MAX_RANGEY = 30
ITERATIONS = 100

BL1 = [[1,1]]
TL1 = [[1,MAX_RANGEY]]
BR1 = [[MAX_RANGEY,1]]
TR1 = [[MAX_RANGEX,MAX_RANGEY]]

INJECT_LIST_BL4 = [[1,1],[1,2],[2,1],[3,1]]
INJECT_LIST_BL2 = [[2,1],[3,1]]
INJECT_LIST_TL2 = [[1,MAX_RANGEY-1],[3,MAX_RANGEY]]


# Cycle through 5-100 densities
for x in range(5,101,5):
    AMOUNT_POINTS = x
    matches_found = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    scores = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]   #db, db1-4, tl2, bl2, bl4, r1, r2, r3, r4
    rotated = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    n_reconstructions = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    total_error = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    with open(f'{str(MAX_RANGEX)}{str(MAX_RANGEY)}{str(AMOUNT_POINTS)}.txt', 'w') as file:
        for q in range(ITERATIONS):

            successes = [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]
            
            # Original database points
            orig_x = list(np.random.randint(low = 1, high=MAX_RANGEX, size=AMOUNT_POINTS))
            orig_y = list(np.random.randint(low = 1, high=MAX_RANGEY, size=AMOUNT_POINTS))

            px_inject = orig_x.copy()
            py_inject = orig_y.copy()
            original = list(zip(orig_x, orig_y))

            inject_list_R1 = []
            inject_list_R2 = []
            inject_list_R3 = []
            inject_list_R4 = []

            # Random points for injection
            for ran in range(1,2):
                inject_list_R1.append([np.random.randint(1, MAX_RANGEX),np.random.randint(1, MAX_RANGEY)])
            for ran in range(1,3):
                inject_list_R2.append([np.random.randint(1, MAX_RANGEX),np.random.randint(1, MAX_RANGEY)])
            for ran in range(1,4):
                inject_list_R3.append([np.random.randint(1, MAX_RANGEX),np.random.randint(1, MAX_RANGEY)])
            for ran in range(1,5):
                inject_list_R4.append([np.random.randint(1, MAX_RANGEX),np.random.randint(1, MAX_RANGEY)])

            # 'inject' the original list'
            injected_TL2 = original.copy()
            injected_TL2.extend(INJECT_LIST_TL2)
            injected_BL2 = original.copy()
            injected_BL2.extend(INJECT_LIST_BL2)
            injected_BL4 = original.copy()
            injected_BL4.extend(INJECT_LIST_BL4)
            injected_R1 = original.copy()
            injected_R1.extend(inject_list_R1)
            injected_R2 = original.copy()
            injected_R2.extend(inject_list_R2)
            injected_R3 = original.copy()
            injected_R3.extend(inject_list_R3)
            injected_R4 = original.copy()
            injected_R4.extend(inject_list_R4)
            injected_BL1 = original.copy()
            injected_BL1.extend(BL1)
            injected_TL1 = original.copy()
            injected_TL1.extend(TL1)
            injected_BR1 = original.copy()
            injected_BR1.extend(BR1)
            injected_TR1 = original.copy()
            injected_TR1.extend(TR1)

            inject_lists = [INJECT_LIST_TL2,INJECT_LIST_BL2,INJECT_LIST_BL4,inject_list_R1,inject_list_R2,inject_list_R3,inject_list_R4,BL1,TL1,BR1,TR1]
            
            # Initialize LEAKER databases
            db = RangeDatabase2D("db1", original, sort = False, center=False, square = False,min_val = (1,1), max_val=(MAX_RANGEX,MAX_RANGEY))
            db_inject_TL2 = RangeDatabase2D("db_inject_TL2", injected_TL2, sort = False, center=False, square = False,min_val = (1,1), max_val=(MAX_RANGEX,MAX_RANGEY))
            db_inject_BL2 = RangeDatabase2D("db_inject_BL2", injected_BL2, sort = False, center=False, square = False,min_val = (1,1), max_val=(MAX_RANGEX,MAX_RANGEY))
            db_inject_BL4 = RangeDatabase2D("db_inject_BL4", injected_BL4, sort = False, center=False, square = False,min_val = (1,1), max_val=(MAX_RANGEX,MAX_RANGEY))
            db_inject_R1 = RangeDatabase2D("db_inject_R1", injected_R1, sort = False, center=False, square = False,min_val = (1,1), max_val=(MAX_RANGEX,MAX_RANGEY))
            db_inject_R2 = RangeDatabase2D("db_inject_R2", injected_R2, sort = False, center=False, square = False,min_val = (1,1), max_val=(MAX_RANGEX,MAX_RANGEY))
            db_inject_R3 = RangeDatabase2D("db_inject_R3", injected_R3, sort = False, center=False, square = False,min_val = (1,1), max_val=(MAX_RANGEX,MAX_RANGEY))
            db_inject_R4 = RangeDatabase2D("db_inject_R4", injected_R4, sort = False, center=False, square = False,min_val = (1,1), max_val=(MAX_RANGEX,MAX_RANGEY))
            db_inject_BL1 = RangeDatabase2D("db_inject_BL1", injected_BL1, sort = False, center=False, square = False,min_val = (1,1), max_val=(MAX_RANGEX,MAX_RANGEY))
            db_inject_TL1 = RangeDatabase2D("db_inject_TL1", injected_TL1, sort = False, center=False, square = False,min_val = (1,1), max_val=(MAX_RANGEX,MAX_RANGEY))
            db_inject_BR1 = RangeDatabase2D("db_inject_BR1", injected_BR1, sort = False, center=False, square = False,min_val = (1,1), max_val=(MAX_RANGEX,MAX_RANGEY))
            db_inject_TR1 = RangeDatabase2D("db_inject_TR1", injected_TR1, sort = False, center=False, square = False,min_val = (1,1), max_val=(MAX_RANGEX,MAX_RANGEY))
            
            
            repeat_list = [db,db,db,db,db,db_inject_TL2,db_inject_BL2,db_inject_BL4,db_inject_R1,db_inject_R2,db_inject_R3,db_inject_R4,db_inject_BL1,db_inject_TL1,db_inject_BR1,db_inject_TR1]
            
            factual_inject = [j.get_numerical_values()[-len(inject_lists[i]):] for i,j in enumerate(repeat_list[5:])]

            # Fix all data types
            factual_inject_temp = []*len(factual_inject)
            for jj, q in enumerate(factual_inject):
                factual_inject_temp.append([])
                for p in q:
                    factual_inject_temp[jj].append(list(p))
            factual_inject=factual_inject_temp
            
             # GET RECONSTRUCTIONS PER METHOD
            for i, d in enumerate(repeat_list):      

                # Get reconstructions
                true_values = d.get_numerical_values()
                att = FMC20(d)
                fdr = att.get_fdr()
                n_reconstructions[i]+=len(fdr)

                # Pick which random dataset points we 'know'
                known_points = [list(random.choices(true_values, k = 1)),list(random.choices(true_values, k = 2)), list(random.choices(true_values, k = 3)),list(random.choices(true_values, k = 4))]
                known_points_new = [] * len(known_points)

                # Fix datatypes
                for ii,p in enumerate(known_points):
                    known_points_new.append([])
                    for q in p:
                        known_points_new[ii].append(list(q))
                known_points= known_points_new
                
                # CYCLE THROUGH RECONSTRUCTIONS
                for k,f in enumerate(fdr):                  
                    
                    reconstruction = f[0].get_numerical_values()
                    
                    point_error = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

                    # CALC ERROR PER POINT
                    for j, p_ in enumerate(reconstruction):              
                        if (p_ not in factual_inject) and i > 4:
                            point_error[i] += math.dist(reconstruction[j], list(true_values)[j])**2
                        else:
                            point_error[i] += math.dist(reconstruction[j], list(true_values)[j])**2
                    
                    # i<5 are non-injected lists
                    if i < 5:
                        temp = point_error[i] / len(reconstruction)
                        total_error[i] += temp

                    if i >= 5:
                        temp = point_error[i]/(len(reconstruction)-len(inject_lists[i-5]))
                        total_error[i]+=temp
                    
                    # Check if correct
                    if reconstruction == true_values:                     
                        if successes[i]==False:
                            scores[i]+=1
                            successes[i]=True
                    
                    # if not, transform all non-original databases (because we have additional information)
                    elif successes[i]==False: 
                        reconstruction = np.array(reconstruction)
                        reconstruction = [list(x) for x in reconstruction] 
                        true_values = [list(x) for x in true_values] 
                        
                        if 1 <= i <= 4:
                            s_points = known_points[i-1]
                            rotated[i-1], scores[i], successes[i], matches_found[i-1] = transform(s_points, reconstruction, rotated[i-1], scores[i], successes[i], true_values, matches_found[i-1])

                        if i >= 5:
                            s_points=factual_inject[i-5]
                            
                            rotated[i-1], scores[i], successes[i], matches_found[i-1] = transform(s_points, reconstruction, rotated[i-1], scores[i], successes[i], true_values, matches_found[i-1])

                        # If it turns out the database was correct, remove the original error of its wrong orientation!
                        if successes[i]:
                            total_error[i]-=temp

        # Write metrics to file
        avg_error = [total_error[i]/n_reconstructions[i] for i in range(16)]
        file.write(f'{scores}\n')
        file.write(f'{rotated}\n')
        file.write(f'{matches_found}\n')

        file.write(f'{total_error}\n')
        file.write(f'{avg_error}\n')
        file.write(f'{n_reconstructions}\n')
