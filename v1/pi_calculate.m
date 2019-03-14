format long;
turns=input('Enter number of iterations for pi series expansion: ');
divisions=input('Enter number of divisions for pi integral estimation: ');

x=[1:4:(turns/2)];
y=(1./x)-(1./(x+2));
series_pi=(sum(y))*4

a=[0:(1/divisions):1];
b=(1./(1+a.^2));
c=b(:,1:20000);
d=b(:,2:20001);
e=((c+d)./2)./divisions;
trap=sum(e);

e=[((1/divisions)/2):(1/divisions):(1-((1/divisions)/2))];
f=(1./(1+(e.^2))).*(1/divisions);
mid=sum(f);
simpson_pi=(((2*mid)+trap)/3)*4