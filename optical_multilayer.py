# Authors: David C. Heson, Jack Liu, Dr. Bill Robertson, June 2022.

# Program to simulate reflection/transmission coefficients across custom multilayers.

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import sys
import copy

def multilayer(n_list, dist, mode = 'rp', step = 1000, option_aw = 'a', e_opt = 'n',
               a1 = 0, a2 = np.pi/2, w_t = 650e-9, w1 = 400e-9, w2 = 750e-9,
               ang = np.pi/3, n_wv_dic = {}, nw_list = [], limit = 1000e-9):

### Parent function, asks for the refraction indeces of all passing mediums, and
### the thickness of the layers the light passes through. Preferably to pass n_list as a
### 1-dimensional np array with complex dtype, if possible. Results will be
### output as radians/meters vs R/T, in the columns 0 and 1 of the matrix that
### the function returns. a1 and a2 give the starting and ending angular positions.
### The w1 and w2 arguments give the wavelength range, in meters. w_t is the constant
### wavelength if the code is ran in angular mode. 'dist' is the list of layer thicknesses.
### option_aw specifies whether the code iterates over wavelengths or angles, the default
### 'a' iterating over angles, and 'w' iterating over wavelengths.
### e_opt being set to 'y' would run the code to output the electric field profile over the
### multilayers with respect to the depth from the top, at a fixed wavelength w_t and
### fixed angle ang.
###
### Advanced:
### If n_wv_dic is specified as ANY non-empty dictionary, the function will automatically
### expect n_wv_dic to contain a series of functions that return the n^2 for the specific
### material, the materials being specified by nw_list which will contain the keys for the
### functions in n_wv_dic, ordered with respect to the layers in dist. Currently, this is
### optimized SPECIFICALLY for wavelength-dependent n's. If you wish to use the
### dictionary feature for n's, n_list can be anything (an empty list [] is a good
### option). Also, make sure that the n's returned are all complex, even if it is +0j


    if (len(n_list) != len(dist) + 2) and (n_wv_dic == 0):
        print("Error, number of layers infered from depths and indexes given does not match.")
        sys.exit()

    destination = np.zeros(shape = (step, 2), dtype = complex)
    nmbr_layers = np.shape(n_list)[0]
    im = complex(0, 1)

    #################################################################################

    def multilayer_matrix(lmbd, theta):

        ### Sub function, it does the actual calculation for the Fresnel coefficients for
        ### fixed angle and wavelength values. It is called by the conditional statements
        ### below it.
        ### Follows the process outlined in DOI:10.1364/AO.29001952

        n_0 = n_list[0]

        C = [
            [1, 0],
            [0, 1]
        ]

        if mode == 'tp' or 'ts':
            t_list = []

        for s in range(0, nmbr_layers - 1):

            if s == 0:
                width = 0
            else:
                width = dist[s-1]

            n1 = n_list[s]
            n2 = n_list[s + 1]

            cost2 = (1 - ((n_0/n2) ** 2) * (np.sin(theta) ** 2)) ** (0.5)
            cost1 = (1 - ((n_0/n1) ** 2) * (np.sin(theta) ** 2)) ** (0.5)

            if mode == 'rp' or mode == 'tp':
                r = (n1 * cost2 - n2 * cost1) / (n1 * cost2 + n2 * cost1)

            elif mode == 'rs' or mode == 'ts':
                r = (n1 * cost1 - n2 * cost2) / (n1 * cost1 + n2 * cost2)

            else:
                print("Error, invalid polarization mode argument.")
                sys.exit()

            if mode == 'tp':
                t_list.append((2 * n1 * cost1) / (n1 * cost2 + n2 * cost1))

            elif mode == 'ts':
                t_list.append((2 * n1 * cost1) / (n1 * cost1 + n2 * cost2))

            delta = complex(n1 * cost1 * 2 * np.pi * width / lmbd, 0)

            factor = [
                [complex(np.exp(-im * delta)), complex(r * np.exp(-im * delta))],
                [complex(r * np.exp(im * delta)), complex(np.exp(im * delta))]
            ]

            C = np.dot(C, factor)

        a = C[0][0]
        c = C[1][0]

        r = c/a

        if mode == 'rp' or mode == 'rs':
            return abs(r) ** 2

        elif mode == 'tp':
            return (np.real((cost2 * n2.conjugate()) / (n_0.conjugate() * np.cos(theta)))
                    * (abs((np.prod(t_list) / a)) ** 2))

        else:
            return (np.real((cost2 * n2) / (n_0 * np.cos(theta)))
                    * (abs((np.prod(t_list) / a)) ** 2))

    ###################################################################################

    def multilayer_aux(lmbd, th, d, nlayer):

        N = len(nlayer)
        wl = lmbd

        dtotal = np.real(sum(d))
        wki = 2 * np.pi / wl

        sinth = []
        costh = []
        wvi = []

        for m in range(0, len(nlayer)):
            sinth.append(np.sin(th) * nlayer[0] / nlayer[m])
            costh.append((1 - (nlayer[0] * np.sin(th)/nlayer[m]) ** 2) ** 0.5)
            wvi.append(wki * nlayer[m])

        rp = []
        tp = []
        rs = []
        ts = []

        for m in range(0, len(nlayer) - 1):

            cost1 = costh[m]
            cost2 = costh[m+1]
            n1 = nlayer[m]
            n2 = nlayer[m+1]

            rp_t = (n2 * cost1 - n1 * cost2) / (n1 * cost2 + n2 * cost1)
            tp_t = (2 * n1 * cost1) / (n1 * cost2 + n2 * cost1)
            rs_t = (n1 * cost1 - n2 * cost2) / (n1 * cost1 + n2 * cost2)
            ts_t = (2 * n1 * cost1) / (n1 * cost1 + n2 * cost2)

            rp.append(rp_t)
            tp.append(tp_t)
            rs.append(rs_t)
            ts.append(ts_t)

        r = []
        t = []
        rss = []
        tss = []

        for i in range(0, len(nlayer) - 1):
            r.append(rp[i])
            t.append(tp[i])
            rss.append(rs[i])
            tss.append(ts[i])

        rr=np.copy(r)
        tt=np.copy(t)
        rrss=np.copy(rss)
        ttss=np.copy(tss)

        for m in range(len(nlayer) - 3, -1, -1):
            decay = 2 * im * d[m+1] * costh[m+1] * wvi[m+1]
            decayr = np.exp(decay)
            decayt = np.exp(im * d[m+1] * costh[m+1] * wvi[m+1])

            topr = r[m] + rr[m+1] * decayr
            topt = t[m] * tt[m+1] * decayt
            bot = 1 + r[m] * rr[m+1] * decayr

            toprs = rss[m] + rrss[m+1] * decayr
            topts = tss[m] * ttss[m+1] * decayt
            bots = 1 + rss[m] * rrss[m+1] * decayr

            rr[m] = topr / bot
            tt[m] = topt / bot
            rrss[m]= toprs / bots
            ttss[m]= topts / bots


        efldi = np.zeros(shape = (len(nlayer), 1), dtype=complex)
        efldr = np.zeros(shape = (len(nlayer), 1), dtype=complex)
        efldis = np.zeros(shape = (len(nlayer), 1), dtype=complex)
        efldrs = np.zeros(shape = (len(nlayer), 1), dtype=complex)

        efldi[0] = 1.
        efldr[0] = rr[0] * efldi[0]

        efldi[len(nlayer) - 1] = tt[0] * efldi[0]
        efldr[len(nlayer) - 1] = 0

        efldis[0] = 1.
        efldrs[0] = rrss[0] * efldis[0]

        efldis[len(nlayer) - 1] = ttss[0] * efldis[0]
        efldrs[len(nlayer) - 1] = 0

        for m in range(0, len(nlayer) - 2):
            decayr = np.exp(2 * im * d[m+1] * costh[m+1] * wvi[m+1])
            decayt2 = np.exp(im * d[m] * costh[m] * wvi[m])

            efldi[m+1] = ((t[m]) / (1 + r[m] * rr[m+1] * decayr)) * (efldi[m] * decayt2)
            efldr[m+1] = ((t[m] * rr[m+1] * decayr) / (1 + r[m] * rr[m+1] * decayr)) * efldi[m] * decayt2

            efldis[m+1] = ((tss[m]) / (1 + rss[m] * rrss[m+1] * decayr)) * (efldis[m] * decayt2)
            efldrs[m+1] = ((tss[m] * rrss[m+1] * decayr) / (1 + rss[m] * rrss[m+1] * decayr)) * efldis[m] * decayt2

        bound = np.zeros(shape = (len(nlayer)+1 , 1))
        bound[0] = 0
        bound[1]=0
        bound[len(nlayer)] = dtotal + 50000

        for m in range (1, len(nlayer) - 1):
            bound[m+1] = bound[m] + np.real(d[m])

        stepz = (dtotal + limit)/step

        j = 1

        etotals = np.zeros(shape = (step, 2))
        etotalss = np.zeros(shape = (step, 2))

        for m in range (0, step):

            z = (m) * stepz
            etotals[m, 0] = z
            etotalss[m, 0] = z

            if z >= bound[j+1]:
                j += 1


            eti = (efldi[j] * np.exp(im * wvi[j] * costh[j] * (z - bound[j])))
            etr = (efldr[j] * np.exp(-1 * im * wvi[j] * costh[j] * (z - bound[j])))
            etis = (efldis[j] * np.exp(im * wvi[j] * costh[j] * (z - bound[j])))
            etrs = (efldrs[j] * np.exp(-1 * im * wvi[j] * costh[j] * (z - bound[j])))


            etotals[m,1] = abs((eti + etr)**2)
            etotalss[m,1] = abs((etis + etrs)**2)

        if (mode == 'rp') or (mode == 'tp'):
            return etotals

        elif (mode == 'rs') or (mode == 'ts'):
            return etotalss

        else:
            print("Error, invalid polarization mode argument.")
            sys.exit()

    ##################################################################################

    ### Conditional Statements:
    ### Here is where the multilayer function is called

    if (e_opt == 'y'):

        empty = [0]
        dist = np.hstack([empty, dist])
        dist = np.hstack([dist, empty])

        destination = multilayer_aux(w_t, ang, dist, n_list)

    elif option_aw == 'a':

        increment = (a2 - a1) / (step)

        for i in range(0, step):

            t = increment * i + a1

            destination[i, 0] = t
            destination[i, 1] = multilayer_matrix(w_t, t)

    elif option_aw == 'w':

        increment = ((w2 - w1) / step)

        if len(n_wv_dic) == 0: ### checks if dictionary is empty, if it is it runs the 'if' block

            for i in range(0, step):

                wv = w1 + i * increment
                destination[i, 0] = wv
                destination[i, 1] = multilayer_matrix(wv, ang)

        elif len(dist) != len(nw_list) - 2:
            print("Error, material list (nw_list) does not match in length with the depth list.")
            sys.exit()

        else: ### ran if the dictionary is NOT empty

            for i in range(0, step):

                n_list = np.zeros(shape = (len(nw_list), 1), dtype = complex)
                wv = w1 + i * increment
                cntr = 0

                for p in nw_list:
                    f = n_wv_dic[p]
                    n_list[cntr, 0] = f(wv)
                    cntr += 1

                destination[i, 0] = wv
                destination[i, 1] = multilayer_matrix(wv, ang)

    else:
        print("Error, invalid angle/wavelength mode argument.")
        sys.exit()

    return destination

###########################################################################################

def bloch_wave(n_list, d, mode = 'rp', step = 1000, option_aw = 'a', a1 = 0,
                 a2 = np.pi/2, w_t = 623.8e-9, w1 = 400e-9, w2 = 750e-9,
                 ang = np.pi/3, roof = 0.98, minimal = 0.6, perc_trav = 0.01, verb = 0):

    ### Function to detect if/where Bloch surface waves occur within a bandgap, given a
    ### variation of angles with a fixed wavelength (option_aw = 'a', a1, a2, w_t), or
    ### vice-versa (option_w = 'w', w1, w2, ang). sens_d determines how many steps through
    ### the given interval have to be above the roof in reflectivity for a bandgap to be
    ### detected, and minimal gives the minimal reflectivity drop to equal the drop with a
    ### Bloch surface wave present. The function returns where the Bloch surface waves
    ### occurs as the wavelength/angle, the rough coordinates defining the gap, and the
    ### minimum in reflectivity which is related to the Bloch Surface Wave.

    diffs = []
    diffsw = []
    sens_d = int(step * perc_trav)
    sim = multilayer(n_list, d, mode, step = step, option_aw = option_aw, a1 = a1, a2 = a2,
                     w_t = w_t, w1 = w1, w2 = w2, ang = ang)
    wv = sim[:,0]
    rat = sim[:,1]

    c = 0
    st = -1
    pos = []

    for i in range(0, step):
        if rat[i] >= roof:
            c += 1
            if c == sens_d:
                st = i - sens_d + 1
                pos.append(st)
        else:
            if st != -1:
                pos.append(i)
                st = -1
            c = 0

    if len(pos) == 0:
        if verb != 0:
            print("No bandgap found within the given region.")
        return False, False, False

    if len(pos) % 2 == 1:
        pos.append(step-1)
    valid = []
    surface_pos = []
    minimum = []

    for i in range(1, len(pos) - 1, 2):
        x1 = pos[i]
        x2 = pos[i+1]
        low = np.amin(rat[x1:x2])
        if (low < minimal) and ((x2 - x1) < (step * 0.002)):
            if verb != 0:
                print("Bloch surface wave between points " + str(x1) + " and " + str(x2) +
                        " with a mininum of " + str(low) + " found.")
            valid.append(pos[i])
            valid.append(pos[i+1])
            minimum.append(low)
            low = np.where(rat[x1:x2] == low)
            low = int(low[0]) + pos[i]
            surface_pos.append(low)
        else:
            continue

    if (len(surface_pos) == 0):
        if verb != 0:
            print("No bandgap with a Bloch surface wave found within the given region.")
        return False, False, False
    else:
        return surface_pos, valid, minimum

##########################################################################################

def swt_calc(n_list, d_list, pol, steps, change = 'a', a_i = 0, a_f = np.pi/2,
             w_c = 623.8e-9, w_i = 400e-9, w_f = 750e-9, a_c = np.pi/3, verb = 0):

    ### Function to calculate the SWT for a specified multilayer.
    ### n_list is the list of refractive indexes for the multilayer, d_list is the list of
    ### widths of the layers within the multilayer, pol is the polarization argument for the
    ### light that passes through the multilayer, change represents over what variable the
    ### code iterates (<a>ngle or <w>avelength). a_i and a_f are initial/final angles, and
    ### w_i and w_f are initial and final wavelengths, and w_c and a_c are the constant
    ### wavelength and angles used for the opposite iterations.
    ### if verb is set to 1, the code will print out the found SWTs, and if it cannot detect
    ### a Bloch Surface Wave for the given layer.

    d_indlist = copy.copy(d_list)
    d_indlist[-1] = d_indlist[-1] + 10e-9

    x, width, low = bloch_wave(n_list, d_list, mode = pol, step = steps, option_aw = change,
                              a1 = a_i, a2 = a_f, w_t = w_c, w1 = w_i, w2 = w_f, ang = a_c,
                              roof = 0.9, minimal = 0.4)
    x_ind, width_ind, low_ind = bloch_wave(n_list, d_indlist, mode = pol, step = steps,
                                    option_aw = change, a1 = a_i, a2 = a_f, w_t = w_c,
                                    w1 = w_i, w2 = w_f, ang = a_c, roof = 0.9, minimal = 0.4)

    if width == False:
        print("Error: no Bloch Surface Wave detected in the given setup.")
        return False

    elif x_ind == False:

        print("Error: a Bloch Surface Wave was detected in the original setup, but the adjusted setup did not exhibit one.")
        print("This error might be fixable by increasing the resolution to 30000 or above.")
        return False

    elif change == 'a':

        diff = abs(x[0] - x_ind[0])
        diff = (diff * (a_f - a_i) / steps) * 180/np.pi
        swt = float(diff)

        if verb == 1:
            print("The degrees per SWT for this design is:", swt)

    elif change == 'w':

        diff = abs(x[0] - x_ind[0])
        diff = (diff * (w_f - w_i) / steps) * 10e9
        swt = float(diff)

        if verb == 1:
            print("The nanometers per SWT for this design is:", swt)

    else:
        print("Error, invalid polarization mode argument.")
        sys.exit()

    return swt

##########################################################################################

def riu_calc(n_list, d_list, pol, steps, change = 'a', a_i = 0, a_f = np.pi/2,
             w_c = 623.8e-9, w_i = 400e-9, w_f = 750e-9, a_c = np.pi/3, verb = 0):

    ### Function to calculate the RIU for a specified multilayer.
    ### n_list is the list of refractive indexes for the multilayer, d_list is the list of
    ### widths of the layers within the multilayer, pol is the polarization argument for the
    ### light that passes through the multilayer, change represents over what variable the
    ### code iterates (<a>ngle or <w>avelength). a_i and a_f are initial/final angles, and
    ### w_i and w_f are initial and final wavelengths, and w_c and a_c are the constant
    ### wavelength and angles used for the opposite iterations.
    ### if verb is set to 1, the code will print out the found RIUs, and if it cannot detect
    ### a Bloch Surface Wave for the given layer.

    n_indlist = copy.copy(n_list)
    n_indlist[-1] = n_indlist[-1] + complex(0.01, 0)

    x, width, low = bloch_wave(n_list, d_list, mode = pol, step = steps, option_aw = change,
                              a1 = a_i, a2 = a_f, w_t = w_c, w1 = w_i, w2 = w_f, ang = a_c,
                              roof = 0.9, minimal = 0.4)
    x_ind, width_ind, low_ind = bloch_wave(n_indlist, d_list, mode = pol, step = steps,
                                    option_aw = change, a1 = a_i, a2 = a_f, w_t = w_c,
                                    w1 = w_i, w2 = w_f, ang = a_c, roof = 0.9, minimal = 0.4)

    if width == False:
        print("Error: no Bloch Surface Wave detected in the given setup.")
        return False

    elif x_ind == False:

        print("Error: a Bloch Surface Wave was detected in the original setup, but the adjusted setup did not exhibit one.")
        print("This error might be fixable by increasing the resolution to 30000 or above.")
        return False

    elif change == 'a':

        diff = abs(x[0] - x_ind[0])
        diff = (diff * (a_f - a_i) / steps) * 180/np.pi
        riu = float(diff / 0.01)

        if verb == 1:
            print("The degrees per RIU for this design is:", riu)

    elif change == 'w':

        diff = abs(x[0] - x_ind[0])
        diff = (diff * (w_f - w_i) / steps) * 10e9
        riu = float(diff / 0.01)

        if verb == 1:
            print("The nanometers per SWT for this design is:", riu)

    else:
        print("Error, invalid polarization mode argument.")
        sys.exit()

    return riu

##########################################################################################

def graph(coord_list, label_list, size = (12, 6), efield = 0, d_set = None):

    ### Basic function to graph results, taking in a list that has numpy arrays of
    ### the data to graph, the first column of each corresponding to x coordinates, and
    ### the second column of which corresponding to y coordinates. The label list gives
    ### what labels each set of coordinates should have.
    ### Has a capability to be used specifically for electrical field simulations,
    ### graphing vertical lines that

    fig, ax = plt.subplots(figsize=(size))
    for i in range(0, len(label_list)):
        curr_list = coord_list[i]
        if efield == 1:
            a = 0
            d_curr = d_set[i]
            for p in d_curr:
                a = p + a
                plt.axvline(x = a, color = "black", linestyle = '--')
        plt.plot(curr_list[:,0], curr_list[:, 1], label = str(label_list[i]))

    plt.legend()
    plt.grid()
    plt.show()

###########################################################################################

def multilayer_explore(n_list, pol, steps, change = 'a',
              a_i = 0, a_f = np.pi/2, w_c = 623.8e-9, w_i = 400e-9, w_f = 750e-9,
              a_c = np.pi/3, def_ext = 400e-9, nm_ext = 500e-9, incr = 1,
              low = 0.4, riu_set = 'no', riu_cond = 3, swt_set = 'no', swt_cond = 0.3, verb = 0):

    ### Function that explores layer width combination for generating band gaps with "deep"
    ### Bloch surface waves, saving the best combinations. n_list is the initial list of
    ### indexes, and pol ('rs' or 'rp') represents the polarization of the light,
    ### steps is the number of steps that the matrix_multilayer function will go through,
    ### change represents whether the angle ('a') or wavelength ('w') are changing.
    ### a_i/a_f are the initial/final angles for changing angle, w_c being the constant
    ### wavelength for those. w_i/w_f are the initial/final wavelengths for changing
    ### wavelength, a_c being the constant angle.
    ### def_ext represents up to how much depth can be added to the defect, and nm_ext
    ### represents how much depth can be added to the non-defect layers. incr is how many
    ### nanometers the code will jump through for each iteration.
    ### The total number of runs should be (def_ext * nm_ext * nm_ext) / incr ** 3.
    ### Returns, in order, a list with lists of widths for the layer, a list with values for
    ### the minimums observed, and the minimum value observed, the indexes matching for
    ### all of them.
    ### If riu_set/swt_set are initialized as 'yes', the code will also filter out multilayers
    ### based on their RIU/SWT values. It will first check if a multilayer achieves the
    ### minimum desired, and then it will check the RIU/SWT minimum conditions.
    ### which are set by riu_cond and swt_cond. If RIU/SWT are turned on and
    ### verbosity is 1 or 2, the code will also give the calculated RIU/SWT
    ### values for the respective multilayers.

    numbr_explr = (round_up((nm_ext*10e9 / incr) * (nm_ext*10e9 / incr) * (def_ext*10e9 / incr)), 0)
    print("A total of " + str(numbr_explr) + " layers expected to be explored.")
    print("Starting incremental parameter exploration...\n")

    d_set = []
    wv_p = []
    dist_n = []
    dist = steps
    n1 = n_list[1]
    n2 = n_list[2]
    nm_ext = int(nm_ext * 10e8)
    def_ext = int(def_ext * 10e8)

    if change == 'a':
        d1 = w_c / (8 * n1 * np.cos(a_i))
        d2 = w_c / (8 * n2 * np.cos(a_i))

    elif change == 'w':
        d1 = w_i / (8 * n1 * np.cos(a_c))
        d2 = w_i / (8 * n2 * np.cos(a_c))
        a_i = np.arcsin()

    else:
        print("Error, invalid angle/wavelength mode argument.")
        sys.exit()

    timer = 0
    timer_d = 0

    init_timer = time.perf_counter()

    for ext_1 in range(0, nm_ext, incr):
        for ext_2 in range(0, nm_ext, incr):

            d = []

            for p in range(0, len(n_list)-2, 2):

                d.append(d1 + ext_1 * 10e-9)
                d.append(d2 + ext_2 * 10e-9)

            d[-1] = d[-1] * 1.2

            for i in range(0, def_ext, incr):

                d[-1] = d[-1] + incr * 10e-9

                x, width, bot = bloch_wave(n_list, d, mode = pol, step = steps, option_aw = change, a1 = a_i,
                            a2 = a_f, w_t = w_c, w1 = w_i, w2 = w_f, ang = a_c, roof = 0.9,
                            minimal = 0.4, verb = 0)

                if (x != False):
                    if bot[0] < low:
                        if riu_set == 'yes':
                            riu_t = riu_calc(n_list, d, pol = pol, steps = steps, change = change,
                                          a_i = a_i, a_f = a_f, w_c = w_c, w_i = w_i, w_f = w_f,
                                          a_c = a_c)
                            if riu_t < riu_cond:
                                continue
                            else:
                                riu_t = round_up(float(riu_t), 3)
                        if swt_set == 'yes':
                            swt_t = swt_calc(n_list, d, pol = pol, steps = steps, change = change,
                                          a_i = a_i, a_f = a_f, w_c = w_c, w_i = w_i, w_f = w_f,
                                          a_c = a_c)
                            if swt_t < swt_cond:
                                continue
                            else:
                                swt_t = round_up(float(swt_t), 3)
                        if change == 'a':
                            for o in range(0, len(x)):
                                x[o] = round_up(float((x[o] / steps) + a_i) * 180/np.pi, 3)
                            for o in range(0, len(width)):
                                width[o] = round_up(float((width[o] / steps) + a_i) * 180/np.pi, 3)
                        elif change == 'w':
                            for o in range(0, len(x)):
                                x[o] = round_up(float((x[o] / steps) + w_i), 3)
                            for o in range(0, len(width)):
                                width[o] = round_up(float((width[o] / steps) + w_i), 3)
                        for o in range(0, len(bot)):
                            bot[o] = round_up(float(bot[o]), 3)
                        d_set.append(copy.copy(d))
                        wv_p.append(copy.copy(x))
                        timer_d += 1
                        if verb == 1:
                            if (riu_set == 'yes') and (swt_set == 'yes'):
                                print("Layer " + str(len(d_set)) +
                                  " appended with a reflectivity minimum of "
                                  + str(bot[0]) + ".RIU value of " + str(riu_t) +
                                  " and SWT value of " + str(swt_t) + ".")
                            elif riu_set == 'yes':
                                print("Layer " + str(len(d_set)) +
                                  " appended with a reflectivity minimum of "
                                  + str(bot[0]) + ".RIU value of " + str(riu_t) + ".")
                            elif swt_set == 'yes':
                                print("Layer " + str(len(d_set)) +
                                  " appended with a reflectivity minimum of "
                                  + str(bot[0]) + ".SWT value of " + str(swt_t) + ".")
                            else:
                                print("Layer " + str(len(d_set)) +
                                  " appended with a reflectivity minimum of "
                                  + str(bot[0]))
                        if verb == 2:
                            if change == 'a':
                                print("Angle of minimum: " + str(x))
                                if (riu_set == 'yes'):
                                    print(str(riu_t) + " degrees per RIU.")
                                if (swt_set == 'yes'):
                                    print(str(swt_t) + " degrees per SWT.")
                            elif change == 'w':
                                print("Wavelength of minimum: " + str(x))
                                if (riu_set == 'yes'):
                                    print(str(riu_t) + " nanometers per RIU.")
                                if (swt_set == 'yes'):
                                    print(str(swt_t) + " nanometers per SWT.")
                            print("Minimum observed: " + str(bot))
                            print(str(timer_d) + "# multilayer setup appended.")
                            real_timer = np.round((time.perf_counter() - init_timer) * 100) / 100
                            print(str(real_timer) + " seconds elapsed.\n")
                del x, width, bot
                timer += 1
                if timer % 50 == 0 and verb > 0:
                    real_timer = np.round((time.perf_counter() - init_timer) * 100) / 100
                    print(str(timer) + " runs in " + str(real_timer) + " seconds.\n")

    real_timer = np.round((time.perf_counter() - init_timer) * 100) / 100
    print("Total time: " + str(real_timer) +  ".")
    print("Total steps: " + str(timer) + ".")
    print("A total of " + str(len(d_set)) + " multilayers obtained.")

    return d_set, wv_p

##########################################################################################

def round_up(n, decimals=0):
    ### boilerplate function to round up numbers for output
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

##########################################################################################
