%% Grid Problem with Q-Learning
close all

%% initialization
grid_size = 10;
input_map = true(grid_size);
input_map(6:10,4) = false;   % Add an obstacle
input_map(6,6:10) = false;   % Add an obstacle
start_coords = [2, 1];
dest_coords = [9, 8];
pits_coords = [2, 4; 4,2; 7,6];

%% Setting up Environment
% set up color map for display
% 1 - white - clear cell
% 2 - black - obstacle
% 3 - red = visited
% 4 - blue  - on list
% 5 - green - start
% 6 - yellow - destination
% 7 - Traced Path
% 8 - Pit

cmap = [1.0000 1.0000 1.0000; ...
        0.2500 0.2500 0.2500; ...
        0.8500 0.3250 0.0980; ...
        0.0000 0.4470 0.7410; ...
        0.4660 0.6740 0.1880; ...
        0.9290 0.6940 0.1250; ...
	    0.6350 0.0780 0.1840;...
        0.4940 0.1840 0.5560];

colormap(cmap);

x = [1 grid_size];
y = [1 grid_size];

global nrows ncols map;
[nrows, ncols] = size(input_map);
map = zeros(nrows, ncols);

map(input_map) = 1;    % Mark free cells
map(~input_map) = 2;   % Mark obstacle cells

% Generate linear indices of start, dest nodes and pits
startNode = sub2ind(size(map), start_coords(1), start_coords(2));
destNode  = sub2ind(size(map), dest_coords(1),  dest_coords(2));
pitNode  = sub2ind(size(map), pits_coords(:,1),pits_coords(:,2));

map(startNode) = 5;
map(destNode)  = 6;
map(pitNode) = 8;

image(x,y,map)
grid on;
axis equal;
axis([0.5 grid_size+0.5 0.5 grid_size+0.5])

%% Learning 
global endEpisode rewardGoal rewardPit rewardStep;

% Rewards
rewardGoal = 10;  % reward for right set of actions
rewardPit = -100;  % penalty for wrong set of actions
rewardStep = -1;  % cost of motion

% Q-Matrix
nActions = 4;  % possible number of actions
nStates = nrows*ncols;
usePrevious=0;

if(usePrevious==1)
    load('gridQLearning.mat');
else
    Q = zeros(nStates,nActions);
end

alpha = 0.9;
gamma = 0.95;
nEpisode = 1000;  % max number of episodes

endEpisode = false;
currentNode = startNode;
iter = 1;
Action_ = [];
prev_Actions = [];
convergence = 0;

while(iter <= nEpisode)
    disp('*************************');
    str = ['Episode ',num2str(iter)]; 
    disp(str);
    
    free_space = find(map==1);
    startNode = free_space(randi(length(free_space),1))
    
    while(endEpisode == false)
       Qmax = max(Q(currentNode,:));
       action = find(Q(currentNode,:)==Qmax,1);
       Action_ = [Action_ action];
       [nextNode,reward] = newState(currentNode,action); 
       Q(currentNode,action) = Q(currentNode,action) + alpha*(reward + gamma*max(Q(nextNode,action)) - Q(currentNode,action));
       map(nextNode) = 4;
       map(currentNode) = 3; 
       image(x,y,map);
       grid on;
       axis equal;
       axis([0.5 grid_size+0.5 0.5 grid_size+0.5])
       pause(0.05);
       currentNode= nextNode;
    end
    disp('*************************'); 
    % reset enviorenment
    map(input_map) = 1;    % Mark free cells
    map(~input_map) = 2;   % Mark obstacle cells
%     startNode = sub2ind(size(map), start_coords(1), start_coords(2));
    destNode  = sub2ind(size(map), dest_coords(1),  dest_coords(2));
    pitNode   = sub2ind(size(map), pits_coords(:,1),pits_coords(:,2));
    map(startNode) = 5;
    map(destNode)  = 6;
    map(pitNode) = 8;
    image(x,y,map);
    axis equal;
    axis([0.5 grid_size+0.5 0.5 grid_size+0.5])
    grid on;
    pause(0.3);
    currentNode = startNode;
    endEpisode = false;
    if(isequal(prev_Actions,Action_))
        convergence = 1;
    end
    prev_Actions = Action_;
    Action_ = [];
    iter = iter +1;
end

function [nextNode,reward] = newState(currentNode,action)
    global endEpisode rewardGoal rewardPit rewardStep nrows ncols map;
    [i, j] = ind2sub([nrows,ncols], currentNode);
    
    switch action
        case 1        % move up
            i = i-1;  
            disp('moved up');
        
        case 2        % move right
            j = j+1; 
            disp('moved right');
        
        case 3        % move left
            j = j-1;
            disp('moved left');
        
        case 4        % move down
            i = i+1;      
            disp('moved down');
    end 
    
    try 
        nextNode = sub2ind([nrows,ncols],i,j);
    catch
        nextNode = currentNode;   
        endEpisode = true;
    end 
        
    switch map(nextNode)
        case 6
            reward = rewardGoal; 
            endEpisode = true;
            
        case 8
            reward = rewardPit;
            endEpisode = true;
            
        case 2
            reward = rewardStep;
            nextNode = currentNode;
            endEpisode = true;    
            
        otherwise
            reward = rewardStep;   
    end
end



