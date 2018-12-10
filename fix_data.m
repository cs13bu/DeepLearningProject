%% Load
dir='C:\Users\cs13\Google Drive\Deep learning\DataGen\Test1\';

radius = 0.22;
degenerate_count = 0;


load([dir,'validation2.mat'])
for i=1:length(cyl1_col)
    
    if cyl_num(i)<1
        cyl1_pos(i,:) = [0,0];
        cyl2_pos(i,:) = [0,0];
        cyl3_pos(i,:) = [0,0];
        degenerate(i) = 0;
    elseif cyl_num(i)<2
        cyl2_pos(i,:) = [0,0];
        cyl3_pos(i,:) = [0,0];
        degenerate(i) = 0;
    elseif cyl_num(i)<3
        cyl3_pos(i,:) = [0,0];
        
        degenerate(i) = 0;
        vec = [cyl1_pos(i,2), - cyl1_pos(i,1)]; vec = vec./norm(vec);
        p1 = cyl1_pos(i,:) + vec*radius;
        p2 = cyl1_pos(i,:) - vec*radius;
        col = check_col(p1,cyl2_pos(i,:),radius);
        col2 = check_col(p2,cyl2_pos(i,:),radius);

        if (col==1)
            degenerate(i) = degenerate(i) + 1;
        end
        if (col2==1) 
            degenerate(i) = degenerate(i) + 1;
        end
        
        vec = [cyl2_pos(i,2), - cyl2_pos(i,1)]; vec = vec./norm(vec);
        p1 = cyl2_pos(i,:) + vec*radius;
        p2 = cyl2_pos(i,:) - vec*radius;
        col = check_col(p1,cyl1_pos(i,:),radius);
        col2 = check_col(p2,cyl1_pos(i,:),radius);

        if (col==1)
            degenerate(i) = degenerate(i) + 1;
        end
        if (col2==1) 
            degenerate(i) = degenerate(i) + 1;
        end
        
    elseif cyl_num(i) == 3
        degenerate(i) = 0;

        vec = [cyl1_pos(i,2), - cyl1_pos(i,1)]; vec = vec./norm(vec);
        p1 = cyl1_pos(i,:) + vec*radius;
        p2 = cyl1_pos(i,:) - vec*radius;
        col1 = check_col(p1,cyl2_pos(i,:),radius);
        col2 = check_col(p1,cyl3_pos(i,:),radius);
        col3 = check_col(p2,cyl2_pos(i,:),radius);
        col4 = check_col(p2,cyl3_pos(i,:),radius);

        if (col1==1 || col2==1)
            degenerate(i) = degenerate(i)+ 1;
        end
        if  (col3==1 || col4==1)
            degenerate(i) = degenerate(i) + 1;
        end
        
        vec = [cyl2_pos(i,2), - cyl2_pos(i,1)]; vec = vec./norm(vec);
        p1 = cyl2_pos(i,:) + vec*radius;
        p2 = cyl2_pos(i,:) - vec*radius;
        col1 = check_col(p1,cyl1_pos(i,:),radius);
        col2 = check_col(p1,cyl3_pos(i,:),radius);
        col3 = check_col(p2,cyl1_pos(i,:),radius);
        col4 = check_col(p2,cyl3_pos(i,:),radius);

        if (col1==1 || col2==1)
            degenerate(i) = degenerate(i) + 1;
        end
        if  (col3==1 || col4==1)
            degenerate(i) = degenerate(i) + 1;
        end
        
        vec = [cyl3_pos(i,2), - cyl3_pos(i,1)]; vec = vec./norm(vec);
        p1 = cyl3_pos(i,:) + vec*radius;
        p2 = cyl3_pos(i,:) - vec*radius;
        col1 = check_col(p1,cyl2_pos(i,:),radius);
        col2 = check_col(p1,cyl1_pos(i,:),radius);
        col3 = check_col(p2,cyl2_pos(i,:),radius);
        col4 = check_col(p2,cyl1_pos(i,:),radius);

        if (col1==1 || col2==1)
            degenerate(i) = degenerate(i) + 1;
        end
        if  (col3==1 || col4==1)
            degenerate(i) = degenerate(i) + 1;
        end
        
    end

end

save([dir,'validation_fixed2.mat'])

 


function col = check_col(vec,circ,radius)
    n = vec ;
    pa = - circ;
    c = n * (dot( pa, n ) / dot( n, n ));
    d = pa - c;
    if sqrt( dot( d, d ) )<=radius && norm(vec)>norm(circ)
        col = true;
    else
        col = false;
    end
end


function h = circle(x,y,r,c)
hold on
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
h = plot(xunit, yunit,'Color',c);
hold off
end

