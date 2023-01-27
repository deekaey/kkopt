
## TODO  ldndc inputs generator

#    def edit_site_xml(self,xml,name,values):
#        positions = [a.end() for a in list(re.finditer(name, xml))]
#        for i in range(len(positions)):
#            act_positions = [a.end() for a in list(re.finditer(name, xml))]
#            startposition=act_positions[i]+2
#            endposition = xml[startposition:].find('"')+startposition
#            new_xml=str(xml[:startposition])+str(values[i])+str(xml[endposition:])
#            xml=new_xml
#        return xml
#
#    def get_linear_regression(self,depths,upper,lower):
#        Dif=lower-upper
#        cum_depths=numpy.cumsum(depths[1:])
#        ratio=cum_depths/numpy.sum(depths[1:])
#        lin_values=list(upper+Dif*ratio)
#        lin_values.insert(0,upper)
#        lin_values.append(lower)
#        return lin_values
#    
#    def write_parameters(self,vector,call=None):
#        if 'dndc' in self.module:
#            print 'dndc'
#            modul_pars=self.pars[numpy.where(self.pars['module']=='scDNDC')]
#        if 'metrx' in self.module:
#            print 'metrx'
#            modul_pars=self.pars[numpy.where(self.pars['module']=='METRX')]
#        
#        #nr_of_other_pars=len(self.pars[numpy.where(self.pars['module']=='scDNDC')]        )
#        #print nr_of_other_pars
#        os.chdir(self.path+os.sep+"projects"+os.sep+"grassland"+os.sep+self.project)
#        out   = open("DE_linden_siteparams"+str(call)+".xml", 'w')    
#        out.write('<?xml version="1.0" ?>\n')
#        out.write('<ldndcsiteparameters>\n')
#        out.write('\n')
#        out.write('    <siteparameters id="0" >\n')
#        out.write('\n')
#        nr_of_cmf_pars=0
#        if 'cmf' in self.module:
#            nr_of_cmf_pars=len(self.pars[numpy.where(self.pars['module']=='cmf')])
#        #print vector
#        for i in range(len(modul_pars)):
#            index=i+self.site_pars+nr_of_cmf_pars+self.nr_of_physiology_pars
#            out.write('    <par name="'+str(modul_pars['name'][i])+'" value="'+str(vector[index])+'" source="orig." />\n')
#            
#        out.write('\n') 
#        out.write('    </siteparameters> \n')
#        out.write('</ldndcsiteparameters>\n')
#        out.close()   
#        
#        siteread = open("DE_linden_site.xml", 'r') 
#        #sitewrite = open("DE_linden_site_new.xml", 'w') 
#        xml=siteread.read()
#        siteread.close()
#        depths=[]
#        positions = [a.end() for a in list(re.finditer('depth', xml))]
#        for i in range(len(positions)):
#            startposition=positions[i]+2
#            endposition = xml[startposition:].find('"')+startposition
#            depths.append(float(xml[startposition:endposition]))
#
#        lin_wc_mins = self.get_linear_regression(depths,vector[0],vector[1])
#        lin_wc_maxs = self.get_linear_regression(depths,vector[2],vector[3])
#        lin_skss    = self.get_linear_regression(depths,vector[4],vector[5])
#        
#        corg=[float(vector[6])]*(len(depths)+1)
#        corg[1]=corg[1]*0.8        
#        corg[2]=corg[2]*0.4
#        corg[3:] = [x*.2 for x in corg[3:]] 
#        norg = [x * .1 for x in corg]
#        
#        bulk=[float(vector[7])]*(len(depths)+1)
#        bulk[1]=bulk[1]+0.2
#        bulk[2]=bulk[2]+0.3
#        bulk[3:] = [x+0.4 for x in bulk[3:]]
#            #alpha=[float(vector[self.site_pars])]*(len(depths)+1)
#            #n=[float(vector[self.site_pars+1])]*(len(depths)+1)
##        lin_wc_maxs=[]
##        for i in range(len(lin_wc_mins)):
##            lin_wc_maxs.append(lin_wc_pluss[i]+lin_wc_mins[i])
#        new_xml=self.edit_site_xml(xml,'sks',lin_skss)
#        new_xml=self.edit_site_xml(new_xml,'corg',corg)
#        new_xml=self.edit_site_xml(new_xml,'norg',norg)
#        new_xml=self.edit_site_xml(new_xml,'bd',bulk)
#        
#        if 'cmf' in self.module:
#            index=self.site_pars+self.nr_of_physiology_pars
#            lin_alpha    = self.get_linear_regression(depths,vector[index],vector[index+1])
#            lin_n        = self.get_linear_regression(depths,vector[index+2],vector[index+3])
#            new_xml=self.edit_site_xml(new_xml,'vangenuchten_alpha',lin_alpha)
#            new_xml=self.edit_site_xml(new_xml,'vangenuchten_n',lin_n)
#        
#        else:
#            new_xml=self.edit_site_xml(new_xml,'wcmin',lin_wc_mins)
#            new_xml=self.edit_site_xml(new_xml,'wcmax',lin_wc_maxs)
#
#        sitewrite = open("DE_linden_site"+str(call)+".xml", 'w') 
#        sitewrite.write(new_xml)
#        sitewrite.close()
#
#        
#        
#        
#        siteread = open("DE_linden_mana.xml", 'r') 
#        #sitewrite = open("DE_linden_site_new.xml", 'w') 
#        xml=siteread.read()
#        siteread.close()
#        names=self.pars[numpy.where(self.pars['module']=='physiology')]['name']
#        for i in range(len(names)):
#            positions = [a.end() for a in list(re.finditer(names[i], xml))]
#            startposition=positions[0]+9
#            endposition = xml[startposition:].find('"')+startposition
#            #print vector[int(self.site_pars)+i]
#            new_xml=str(xml[:startposition])+str(vector[self.site_pars+i])+str(xml[endposition:])
#            #new_xml=str(xml[:startposition])+str(2)+str(xml[endposition:])
#            xml=new_xml
#
#        positions = [a.end() for a in list(re.finditer('STRAW', xml))]
#        startposition=positions[0]+9
#        endposition = xml[startposition:].find('"')+startposition
#        #print vector[int(self.site_pars)+i]
#        new_xml=str(xml[:startposition])+str(0.95-vector[self.site_pars])+str(xml[endposition:])
#        #new_xml=str(xml[:startposition])+str(2)+str(xml[endposition:])
#        xml=new_xml        
#        sitewrite = open("DE_linden_mana"+str(call)+".xml", 'w') 
#        sitewrite.write(xml)
#        sitewrite.close()
#        os.chdir(self.owd)
#
#        os.chdir(self.path+os.sep+"projects"+os.sep+"grassland"+os.sep+"DE_linden"+os.sep+"DE_linden_"+self.module)
#        lines = open("DE_linden_"+self.module+".xml").readlines()
#        out   = open("DE_linden_"+self.module+str(call)+".xml", 'w')
#        for i in range(len(lines)):    
#            write=False
#            if i==3:
#                if self.module == 'dndc':  
#                    out.write('    <schedule  time="'+str(self.analysestart.year)+'-'+str(self.analysestart.month)+'-'+str(self.analysestart.day)+'/1 -> +'+str(self.rundays)+'" />\n')
#                else:
#                    out.write('    <schedule  time="'+str(self.analysestart.year)+'-'+str(self.analysestart.month)+'-'+str(self.analysestart.day)+'/24 -> +'+str(self.rundays)+'" />\n')
#                write=True
#            if i==8:
#                out.write('            <site  source="../DE_linden_site'+str(call)+'.xml" />\n')
#                write=True
#            
#            if i==11:
#                out.write('            <event  source="../DE_linden_mana'+str(call)+'.xml" />\n')
#                write=True                
#
#            if i==12:
#                out.write('            <siteparameters  source="../DE_linden_siteparams'+str(call)+'.xml" />\n')
#                write=True
#                
#            if i==21:
#                if 'metrx' in self.module:
#                    out.write('        <sinks sinkprefix="grassland/DE_linden/DE_linden_'+self.module+'/DE_linden_'+self.module+'_output'+str(call)+'/DE_linden_'+self.module+'_" > \n')
#                else:                   
#                    out.write('        <sinks sinkprefix="grassland/DE_linden/DE_linden_'+self.module+'/DE_linden_'+self.module+'_output'+str(call)+'/DE_linden_'+self.module+'_" /> \n')
#                write=True
#                
#            if not write==True:
#                out.write(lines[i])
#            
#        out.close()
#        os.chdir(self.owd )
#        
#    def remove_files(self,call=None):
#        os.chdir(self.path+os.sep+"projects"+os.sep+"grassland"+os.sep+self.project)                
#        try:
#            os.remove("DE_linden_site"+str(call)+".xml")
#            os.remove("DE_linden_mana"+str(call)+".xml")
#            os.remove("DE_linden_siteparams"+str(call)+".xml")
#            os.chdir(self.owd )
#            os.chdir(self.path+os.sep+"projects"+os.sep+"grassland"+os.sep+self.project+os.sep+self.project+"_"+self.module)
#            os.remove("DE_linden_"+self.module+str(call)+".xml")
#            shutil.rmtree('DE_linden_'+self.module+'_output'+str(call))
#            os.chdir(self.owd )
#        
#        except OSError:
#            print str(sys.exc_info()[1])
#            os.chdir(self.owd)


