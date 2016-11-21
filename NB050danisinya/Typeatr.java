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
public abstract class Typeatr {
    public abstract double getProbability(int d);
    public abstract void addValue(double data, double weight);
    public abstract double getProbability(double data); 
}
