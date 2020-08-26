function [fwd_mesh,pj_error] = reconstruct_cw_OGCV(fwd_fn,...
    data_fn,...
    iteration,...
    lambda,...
    output_fn,...
    filter_n)

% [fwd_mesh,pj_error] = reconstruct_stnd_cw(fwd_mesh,...
%                                        recon_basis,...
%                                        data_fn,...
%                                        iteration,...
%                                        lambda,...
%                                        output_fn,...
%                                        filter_n)
%
% CW Reconstruction program for standard meshes
%
% fwd_mesh is the input mesh (variable or filename)
% recon_basis is the reconstruction basis (pixel basis or mesh filename)
% data_fn is the boundary data (variable or filename)
% iteration is the max number of iterations
% lambda is the initial regularization value
% output_fn is the root output filename
% filter_n is the number of mean filters




% set modulation frequency to zero.
frequency = 0;
% iteration =1;lambda=10;
tic;
%****************************************
% If not a workspace variable, load mesh
if ischar(fwd_fn)== 1
    fwd_mesh = load_mesh(fwd_fn);
end


%*******************************************************
% read data - This is the calibrated experimental data or simulated data
anom = load(data_fn);
anom = log(anom(:,1));


% Initiate projection error
pj_error = [];
%*******************************************************
% Initiate log file
fid_log = fopen([output_fn '.log'],'w');
% fprintf(fid_log,'Forward Mesh   = %s\n',fwd_mesh.name);
% if ischar(recon_basis)
%     fprintf(fid_log,'Basis          = %s\n',recon_basis);
% end
fprintf(fid_log,'Frequency      = %f MHz\n',frequency);
if ischar(data_fn) ~= 0
    fprintf(fid_log,'Data File      = %s\n',data_fn);
end

% fprintf(fid_log,'Initial Regularization  = %d\n',lambda);

fprintf(fid_log,'Filter         = %d\n',filter_n);
fprintf(fid_log,'Output Files   = %s_mua.sol\n',output_fn);
fprintf(fid_log,'               = %s_mus.sol **CW recon only**\n',output_fn);
fprintf(fid_log,'Initial Guess mua = %d\n',fwd_mesh.mua(1));


for it = 1 : iteration
    
    % Calculate jacobian
    [J,data]=jacobian_stnd(fwd_mesh,frequency);
    
    % Set jacobian as Phase and Amplitude part instead of complex
    J = J.complete;
    [nrow ncol] = size(J);
    % Read reference data
    clear ref;
    ref = log(data.amplitude);
    
    data_diff = (anom-ref);
    
    pj_error = [pj_error sum(abs(data_diff.^2))];
    
    disp('---------------------------------');
    disp(['Iteration Number          = ' num2str(it)]);
    disp(['Projection error          = ' num2str(pj_error(end))]);
    
    fprintf(fid_log,'---------------------------------\n');
    fprintf(fid_log,'Iteration Number          = %d\n',it);
    fprintf(fid_log,'Projection error          = %f\n',pj_error(end));
    
    
    if it ~= 1
        p = (pj_error(end-1)-pj_error(end))*100/pj_error(end-1);
%         p_stop = 0.02*pj_error(1)
        
        disp(['Projection error change   = ' num2str(p) '%']);
        fprintf(fid_log,'Projection error change   = %f %%\n',p);
        
        if  ( (p <= 2) )%|| (pj_error(end)  <= p_stop) 
            disp('---------------------------------');
            disp('STOPPING CRITERIA REACHED');
            fprintf(fid_log,'---------------------------------\n');
            fprintf(fid_log,'STOPPING CRITERIA REACHED\n');
            break
        end
    end
    
      % Normalize Jacobian wrt optical values
            J = J*diag([fwd_mesh.mua]);
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % build hessian    
    %Hess = zeros(ncol);
    %Hess = (J'*J);

    if it ==1
  a= 0; b = 1000;
    end
    [U S V ] = svd(J);s = diag(S);
    bhat = U'*data_diff;
    lambda = fminbnd('TikGCVfun', a, b, ...
        optimset('TolX', 1e-16),...
        bhat,s);
%      b= lambda;
%         f_gcv= Tikhonov(U, s, V, bnoise, alpha_tikgcv); % tikhonov restoration
        



     [~,ncol]=size(J);
        Hess = zeros(ncol);
        Hess = (J'*J);
%             [nrow,ncol]=size(J);
%         Hess = zeros(nrow);
%         Hess = (J*J');
        
        % Add regularization
%         if it ~= 1
%             lambda = lambda./10^0.25;
%         end
        lambda
        reg = eye(ncol).*( lambda); %reg = single(reg);
        disp(['Amp Regularization        = ' num2str(reg(1,1))]);
        fprintf(fid_log,'Amp Regularization        = %f\n',reg(1,1));
        
        Hess = Hess+reg;
        
        
        % Calculate update
         foo = Hess\(J'*data_diff);
%                foo = J'*(Hess\data_diff);
        foo = foo.*[fwd_mesh.mua];
        
        % Update values
        
        fwd_mesh.mua = fwd_mesh.mua + foo;
        fwd_mesh.kappa = (1./(3.*(fwd_mesh.mus+fwd_mesh.mua)));
        
    
% % % % % % % % % % % % % % % % % % % % % %

%     
    
    clear  Hess Hess_norm tdmp data_diff G
    
    
    
    % We dont like -ve mua or mus! so if this happens, terminate
    if (any(fwd_mesh.mua<0) | any(fwd_mesh.mus<0))
        disp('---------------------------------');
        disp('-ve mua calculated...');
        fprintf(fid_log,'---------------------------------\n');
        fprintf(fid_log,'STOPPING CRITERIA REACHED\n');
         break
    end
    
    % Filtering if needed!
    if filter_n > 1
        fwd_mesh = mean_filter(fwd_mesh,abs(filter_n));
    elseif filter_n < 0
        fwd_mesh = median_filter(fwd_mesh,abs(filter_n));
    end
    
    if it == 1
        fid = fopen([output_fn '_mua.sol'],'w');
    else
        fid = fopen([output_fn '_mua.sol'],'a');
    end
    fprintf(fid,'solution %g ',it);
    fprintf(fid,'-size=%g ',length(fwd_mesh.nodes));
    fprintf(fid,'-components=1 ');
    fprintf(fid,'-type=nodal\n');
    fprintf(fid,'%f ',fwd_mesh.mua);
    fprintf(fid,'\n');
    fclose(fid);
    
    if it == 1
        fid = fopen([output_fn '_mus.sol'],'w');
    else
        fid = fopen([output_fn '_mus.sol'],'a');
    end
    fprintf(fid,'solution %g ',it);
    fprintf(fid,'-size=%g ',length(fwd_mesh.nodes));
    fprintf(fid,'-components=1 ');
    fprintf(fid,'-type=nodal\n');
    fprintf(fid,'%f ',fwd_mesh.mus);
    fprintf(fid,'\n');
    fclose(fid);
end
% hold off;

% close log file!
time = toc;
fprintf(fid_log,'Computation Time = %f\n',time);
fclose(fid_log);

