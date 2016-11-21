/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tucil;


/**
 *
 * @author mac
 */
public class Discreteatr extends Typeatr{
    
    double[] count;
    double sumcount;
    
    Discreteatr(int numsym, boolean start) {
        count = new double [numsym];
        sumcount = 0;
        if (start) {
          for(int i = 0; i < numsym; i++) {
            count[i] = 1;
          }
          sumcount = (double)numsym;
        }
    }

    @Override
    public double getProbability(int d) 
    {
        if (sumcount == 0) 
        {
            return 0;
        }
            return (double)count[d] / sumcount;
    }
    
    @Override
    public void addValue(double data, double weight) 
    {
        count[(int)data] += weight;
        sumcount += weight;
    }

    @Override
    public double getProbability(double d) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
