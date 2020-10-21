import cv2
import numpy as np
import operator

def training():
    conturearea = 100
    conturehight = 30
    conturewidth = 20


    trainimage = cv2.imread('training_chars.png')
    trainingimggray=cv2.cvtColor(trainimage,cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(trainingimggray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(imgBlurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    contures , res = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    '''print(thresh[0])
    cv2.imshow('training image ',thresh)'''


    charlable = []
    flatternIMG = []

    ValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                         ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                         ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                         ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z'),ord('a'), ord('b'), ord('c'), ord('d'),
                         ord('e'), ord('f'), ord('g'), ord('h'), ord('i'), ord('j'),
                         ord('k'), ord('l'), ord('m'), ord('n'), ord('o'), ord('p'), ord('q'), ord('r'), ord('s'), ord('t'),
                         ord('u'), ord('v'), ord('w'), ord('x'), ord('y'), ord('z')]

    for conture in contures:
        if cv2.contourArea(conture)>conturearea:

            x,y,w,h = cv2.boundingRect(conture)
            cv2.rectangle(trainimage,(x,y),(x+w,y+h),(0,0,255), 2 )

            imgROI= thresh[y:y+h,x:x+w]
            imgROIresize = cv2.resize(imgROI,(conturewidth,conturehight))

            cv2.imshow('training', trainimage)
            cv2.imshow('roi', imgROI)
            cv2.imshow('resize', imgROIresize)


            keyinput = cv2.waitKey(0)
            if keyinput == 27:
                break
            elif keyinput in ValidChars:
                charlable.append(keyinput)
                imgROIresize = imgROIresize.flatten()
                flatternIMG.append(imgROIresize)

    flatternIMG = np.array(flatternIMG, dtype=np.float32)
    charlable = np.array(charlable,dtype=np.float32)

    print ("\n\ntraining complete !!\n")

    np.savetxt("classifications.txt", charlable)           # write flattened images to file
    np.savetxt("flattened_images.txt", flatternIMG)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def testing ():

    conturearea = 100
    conturewidth = 20
    contureheight = 30


    allContoursWithData=[]
    validContoursWithData =[]

    flatimage = []

    classificationdata = np.loadtxt("classifications.txt", np.float32)
    flattenimagedata = np.loadtxt("flattened_images.txt", np.float32)


    KNN = cv2.ml.KNearest_create()
    KNN.train(flattenimagedata,cv2.ml.ROW_SAMPLE,classificationdata)

    testimg = cv2.imread('pgga.png')
    testimggray = cv2.cvtColor(testimg,cv2.COLOR_BGR2GRAY)
    blurr = cv2.GaussianBlur(testimggray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blurr,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

    contures,res = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)



    for conture in contures:

        x,y,w,h = cv2.boundingRect(conture)
        allContoursWithData.append([x,y,w,h])


        allContoursWithData.sort(key=lambda i:(i[1]//100,i[0])) #sorting 2D array up to down and left to right

    '''allContoursWithData.sort(key=lambda y: y[1])

    refinedconture1=[]
    refinedconture2 =[]
    refine =[]
    for i in range(0,len(allContoursWithData)):
        if abs(allContoursWithData[0][1]-allContoursWithData[i][1])<5 :
            refinedconture1.append(allContoursWithData[i])

        elif abs(allContoursWithData[0][1]-allContoursWithData[i][1])>5:
            refinedconture2.append(allContoursWithData[i])


    refinedconture1.sort(key=lambda y: y[0])
    refinedconture2.sort(key=lambda y: y[0])
    refine.extend(refinedconture1)
    refine.extend(refinedconture2)
    print(refine)'''


    for x,y,w,h in allContoursWithData:
        cv2.rectangle(testimg,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow('test',testimg)

        testcrop = thresh[y:y+h,x:x+w]
        cv2.imshow('croped test ', testcrop)
        testcrop=cv2.resize(testcrop,(conturewidth,contureheight))
        testcrop = testcrop.flatten()

        flatimage.append(testcrop)





    test_cells = np.array(flatimage, dtype=np.float32)

    ret, result, neighbours, dist = KNN.findNearest(test_cells, k=3)
    detectedstr = ''
    for i in range(len(result)):
        detectedstr=detectedstr+str(chr(int(result[i][0])))

    print(detectedstr)
    cv2.waitKey(0)





if __name__ == '__main__':


    testing()