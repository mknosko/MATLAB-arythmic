clear
load dataarytmiasrdca

datainnet=NDATA';
dataoutnet=parseClassify(typ_ochorenia);

%  vytvorenie struktury siete
pocet_neuronov=20;  % definujte pocet neuronov v skrytej vstve
net = patternnet(pocet_neuronov);

skupina1 = (1:88);
skupina2 = (89:176);
skupina3 = (177:264);
skupina4 = (265:352);
skupina5 = (353:442);

% net.divideFcn='dividerand';
% net.divideParam.trainRatio=0.6;
% net.divideParam.valRatio=0.2;
% net.divideParam.testRatio=0.2;

net.divideFcn = 'divideind';
% Indexove rozdelenie skupin 1,3,5; 4; 2;
% net.divideParam.trainInd = [1:90, 167:256, 333:442];
% net.divideParam.valInd = (256:332); 
% net.divideParam.testInd = (91:166);

% Indexove rozdelenie skupin 1,2,3; 4; 5;
% net.divideParam.trainInd = [skupina1, skupina2, skupina3];
% net.divideParam.valInd = skupina4; 
% net.divideParam.testInd = skupina5;

% % Indexove rozdelenie skupin 3,4,5; 1; 2;
% net.divideParam.trainInd = [skupina3, skupina4, skupina5];
% net.divideParam.valInd = skupina1; 
% net.divideParam.testInd = skupina2;

% Indexove rozdelenie skupin 1,2,4; 3; 5;
net.divideParam.trainInd = [skupina1, skupina2, skupina5];
net.divideParam.valInd = skupina3; 
net.divideParam.testInd = skupina4;

% nastavenie parametrov trenovania
% definujte parametre trenovania siete
net.trainParam.goal = 1e-10;       % ukoncovacia podmienka na chybu.
net.trainParam.show = 10;           % frekvencia zobrazovania chyby
net.trainParam.epochs = 500;    % maximalny pocet trenovacich epoch.
net.trainParam.max_fail = 250;
net.trainParam.min_grad = 1e-10;

counter(typ_ochorenia)'

% trenovanie neuronovej siete
net = train(net,datainnet,dataoutnet);
% simulacia vystupu NS pre trenovacie data
% testovanie NS
outnetsim = sim(net,datainnet);

% chyba NS a dat
err=(outnetsim-dataoutnet);

% plot confusion
y=net(datainnet);
plotconfusion(dataoutnet, y);