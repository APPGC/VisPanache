from reference_diagnostics import ReferenceConfig,assess_reference
import numpy as np
c=ReferenceConfig(row_start=0,row_stop=2,control_start=3,control_stop=5)
a=np.full((2,5,3),.6,np.float32);b=np.full_like(a,.8);m=np.zeros((2,5),bool);m[:,1]=True;a[:,1]=.7;w=np.ones((2,5),np.float32)
r=assess_reference(a,b,m,w,c);assert np.all((r['case_warning_flags'][:,1]&12)==12);assert not r['case_warning_flags'][~m].any();assert np.all(r['archived_reference_contrast'][:,1]<0);assert np.all(r['same_row_control_contrast'][:,1]>0)
a[:,1]=.9;r=assess_reference(a,b,m,w,c);assert np.all((r['case_warning_flags'][:,1]&1)==1);assert not (r['case_warning_flags']&48).any()
a[:,1]=1;r=assess_reference(a,b,m,w,c);assert np.all((r['case_warning_flags'][:,1]&16)==16)
m[:,3:]=True;r=assess_reference(a,b,m,w,c);assert np.all((r['case_warning_flags'][:,1]&64)==64);assert not r['reference_comparison_mask'].any();assert np.isnan(r['same_row_control_contrast']).all()
print('PASS: opposite reference signs, mask scoping, bright unsaturated signal, clipping and contaminated-control rejection')
