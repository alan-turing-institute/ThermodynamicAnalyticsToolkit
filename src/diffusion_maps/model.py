import numpy as np


class Model():

    def __init__(self,potentialFlag, hDimer=1.0, wDimer=1.0, alpha=0.0):

          self.hDimer=hDimer
          self.wDimer=wDimer
          #for double-well in 2d with 2 different heights in x and y axis
          self.hDimerModif=2.0*self.hDimer
          self.wDimerModif=self.wDimer

        #   print('hDimer = '+repr(self.hDimer))
        #   print('wDimer = '+repr(self.wDimer))
        #   print('hDimer2 = '+repr(self.hDimerModif))
        #   print('wDimer2 = '+repr(self.wDimerModif))

          self.alpha=alpha
          self.potentialFlag=potentialFlag

          if potentialFlag=='HO':
              self.potential=self.HO
              self.force=self.fHO
          if potentialFlag=='DW':
              self.potential=self.DW
              self.force=self.fDW
          if potentialFlag=='DW_X_HO_Y':
              self.potential=self.DW_X_HO_Y
              self.force=self.fDW_X_HO_Y
          if potentialFlag=='DW2D':
              self.potential=self.DW2d
              self.force=self.fDW2d
          if potentialFlag=='HO_X_HO_Y':
              self.potential=self.HO_X_HO_Y
              self.force=self.fHO_X_HO_Y
          if potentialFlag=='DW_2D_HO_1D':
              self.potential=self.DW_2D_HO_1D
              self.force=self.fDW_2D_HO_1D
          if potentialFlag=='TRIPLE_WELL':
             self.potential=self.TRIPLE_WELL
             self.force=self.fTRIPLE_WELL
          if potentialFlag=='LIFTED_DW2D':
             self.potential=self.LIFTED_DW2D
             self.force=self.fLIFTED_DW2D
          if potentialFlag=='MULTIPLE_WELL':
             self.potential=self.MULTIPLE_WELL
             self.force=self.fMULTIPLE_WELL
          if potentialFlag=='BANANA':
             self.potential=self.BANANA
             self.force=self.fBANANA
          if potentialFlag=='BROWNIAN':
             self.potential=self.BROWNIAN
             self.force=self.fBROWNIAN
          if potentialFlag=='CIRCULAR':
             self.potential=self.CIRCULAR
             self.force=self.fCIRCULAR
          if potentialFlag=='SINCOS':
             self.potential=self.SINCOS
             self.force=self.fSINCOS
          if potentialFlag=='DW1_X_DW2_Y':
             self.potential=self.DW1_X_DW2_Y
             self.force=self.fDW1_X_DW2_Y

    def HO(self,x):
        return 0.5*self.hDimer*np.linalg.norm(x)**2

    def fHO(self,x):
        #f=np.zeros(x.shape)
        return self.hDimer*x

    def DW(self, x):
        return self.hDimer*(np.linalg.norm(x)**2-self.wDimer)**2

    def fDW(self, x):
        return self.hDimer*4*(x**2-self.wDimer)*x

    # doubled barrier height
    def DW_ModifyBarrier(self, x):
        return self.hDimerModif*(np.linalg.norm(x)**2-self.wDimerModif)**2

    def fDW_ModifyBarrier(self, x):
        return self.hDimerModif*4.0*(x**2-self.wDimerModif)*x
#######
    def DW_X_HO_Y(self, x):
        return self.DW(x[0])+self.HO(x[1]) + 0.1*x[0]*x[1]

    def fDW_X_HO_Y(self, x):
        f=np.zeros(x.shape)
        f[0]=self.hDimer*4*(x[0]**2-self.wDimer)*x[0] +0.1*x[1]
        f[1]=x[1] +0.1*x[0]

        return f


    def DW2d(self, x):
        w=self.wDimer
        h=self.hDimer
        return (1/6.0)*(4*(-x[0]**2-x[1]**2+w)**2+2*h*(x[0]**2-2)**2+((x[0]+x[1])**2-w)**2+((x[0]-x[1])**2-w)**2)

    def fDW2d(self, x):
        f=np.zeros(x.size)
        w=self.wDimer
        h=self.hDimer

        f[0]=4.0*x[0]**3+(20.0/3.0)*x[0]*x[1]**2-4.0*w*x[0]+(4.0/3.0)*h*x[0]**3-(8.0/3.0)*h*x[0]
        f[1]=(20.0/3.0)*x[0]**2*x[1]+4*x[1]**3-4.0*w*x[1]

        return f

    def HO_X_HO_Y(self, x):
        return self.hDimer*self.HO(x[0])+self.wDimer*self.HO(x[1])

    def fHO_X_HO_Y(self, x):
        f=np.zeros(x.size)

        f[0]= self.hDimer*x[0]
        f[1]= self.wDimer*x[1]

        return f

    def DW_2D_HO_1D(self, x):
        return self.DW(x[0])+self.DW(x[1])+self.HO(x[2])

    def fDW_2D_HO_1D(self, x):
        f=np.zeros(x.size)
        h=self.hDimer
        w=self.wDimer

        f[0]=h*4*(x[0]**2-w)*x[0]
        f[1]=h*4*(x[1]**2-w)*x[1]
        f[2]=x[2]

        return f

    def TRIPLE_WELL(self, x):
        return  3*np.exp(-x[0]**2-(x[1]-1/3.0)**2)-3*np.exp(-x[0]**2-(x[1]-5/3.0)**2)-5.*np.exp(-(x[0]-1)**2-x[1]**2)-5.0*np.exp(-(x[0]+1)**2-x[1]**2)+.2*x[0]**4+.2*(x[1]-1/3.0)**4

    def fTRIPLE_WELL(self, x):
        f=np.zeros(x.size)

        f[0]=-6*x[0]*np.exp(-x[0]**2-(x[1]-1/3)**2)+6*x[0]*np.exp(-x[0]**2-(x[1]-5/float(3))**2)-(5*(-2*x[0]+2))*np.exp(-(x[0]-1)**2-
                                                                                                                         x[1]**2)-(5*(-2*x[0]-2))*np.exp(-(x[0]+1)**2-x[1]**2)+.8*x[0]**3
        f[1]=(3*(-2*x[1]+2/float(3)))*np.exp(-x[0]**2-(x[1]-1/float(3))**2)-(3*(-2*x[1]+10/float(3)))*np.exp(-x[0]**2-(x[1]-5/float(3))**2)+10*x[1]*np.exp(-(x[0]-1)**2-x[1]**2)+10*x[1]*np.exp(-(x[0]+1)**2-x[1]**2)+.8*(x[1]-1/float(3))**3
        return f

    def LIFTED_DW2D(self, x):
            return self.DW2d(x)+ (x[0]-1)**2

    def fLIFTED_DW2D(self, x):
        f=np.zeros(x.size)
        f=self.fDW2d(x)
        f[0]=f[0]+ 2.0*(x[0]-1)
        return f

    def MULTIPLE_WELL(self, x):
        return self.multipleWell(x[0]) + self.multipleWell(x[1]) + 0.01*x[0]*x[1]

    def fMULTIPLE_WELL(self, x):
        f=np.zeros(x.size)
        f[0]=self.diffMultipleWell(x[0]) +0.01*x[1]
        f[1]=self.diffMultipleWell(x[1]) +0.01*x[0]

        return f

    def multipleWell(self, x):
        h=self.hDimer
        w=self.wDimer
        return h/float(1.0/(x-w)**2 + 1.0/(x+w)**2 + 1.0/(x-2*w)**2+1.0/(x+2*w)**2)


    def diffMultipleWell(self, x):
        h=self.hDimer
        w=self.wDimer
        return -h*x*(204*w**8-115*w**6*x**2+21*w**4*x**4-2*x**8)*(w**2-x**2)*(4*w**2-x**2)/(20*w**6+w**4*x**2-5*w**2*x**4+2*x**6)**2


    def BANANA(self, x):
        return (0.5*((0.01*x[0]**2)+(x[1]+0.1*x[0]**2-10)**2))

    def fBANANA(self, x):
        f=np.zeros(x.size)
        f[0]=-1.990*x[0]+0.20*x[0]*x[1]+0.020*x[0]**3
        f[1]=x[1]+0.10*x[0]**2-10.0

        return f

    def BROWNIAN(self, x):
        return 0.0

    def fBROWNIAN(self, x):
        return np.zeros(x.size)

    def CIRCULAR(self, x):
        return np.cos(self.hDimer*np.arctan2(x[1], x[0]))+10.0*(np.sqrt(x[0]**2+x[1]**2)-1.0)**2

    def fCIRCULAR(self, x):
        f=np.zeros(x.size)

        t1=np.sin(self.hDimer*np.arctan2(x[1], x[0]))/(x[0]**2*(1.0+x[1]**2/x[0]))
        t2=(20.0*(np.sqrt(x[0]**2+x[1]**2)-1.0) )/(np.sqrt(x[0]**2+x[1]**2))

        f[0]=self.hDimer*x[1]/x[0]*t1 + x[0]*t2
        f[1]=-self.hDimer*t1+t2*x[1]

        return f


    def SINCOS(self, x):
        w=self.wDimer
        h=self.hDimer
        return h*np.cos(x[0])*np.sin(x[1])+w*(x[0]**2+x[1]**2)

    def fSINCOS(self, x):
        f=np.zeros(x.size)
        w=self.wDimer
        h=self.hDimer

        f[0]=-h*np.sin(x[0])*np.sin(x[1])+2*w*x[0]
        f[1]= h*np.cos(x[0])*np.cos(x[1])+2*w*x[1]

        return f

    def DW1_X_DW2_Y(self, x):
        return self.DW_ModifyBarrier(x[0]) + self.DW(x[1])

    def fDW1_X_DW2_Y(self, x):
        f=np.zeros(x.size)

        f[0]= self.fDW_ModifyBarrier(x[0])
        f[1]= self.fDW(x[1])

        return f
