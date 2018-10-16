function [data, source_signal, Asignal] = gen_data(T, D, N, K, snr, ssnr, samesignal, samenoise, pink, distr)
% generate data as linear mixture of K correlated components and background noise
%
% T: number of time samples
% D: number of sensors
% N: number of subjects/viewings
% K: number of correlated dimensions
% snr: snr in [0, 1]
% ssnr: spatially uncorrelated to correlated noise ratio in [0 1]
% samesignal: 1/0, indicates same or different signal correlation structure
% samenoise:  1/0, indicates same or different noise correlation structure
% distr: distribution of signal and noise sources
%   either 'Gauss' (default), 'Chi2' or 'Bernoulli'
%   in the latter case, observations are also dichotomized.
%
% Stefan Haufe, 2017

if length(K) > 1
  target_ISC = K;
  K = length(K);
else
  target_ISC = (K:-1:1)/K;
end
alphas = sqrt(1+target_ISC).*sqrt(target_ISC)./sqrt(1-target_ISC.^2);
betas = ones(size(alphas));
ininf = isinf(alphas);
alphas(ininf) = 1;
betas(ininf) = 0;
  
if nargin < 6
  ssnr = 0;
end

if nargin < 8
  samesignal = 1;
  samenoise = 1;
end

if nargin < 9
  pink = 1;
end

if nargin < 10
  distr = 'Gauss';
end

sensor_signal = zeros(T, D, N); 

if K == 0
  source_signal = [];
else
  % time course of the correlated activity = signal 
  % at this point white or pink Gaussian noise
  % assumed to be identical for all subject/viewings
  % this may be relaxed
  if pink
    source_signal = mkpinknoise(T, K);
  else
    source_signal = randn(T, K);
  end
  
  % make signal chi2 or Bernoulli distributed
  switch distr
    case 'Chi2'
      source_signal = source_signal.^2;
%     case 'Bernoulli'
%       source_signal = sign(source_signal);
  end
  
  source_signal = repmat(source_signal, 1, 1, N);
  
  source_signal = zscore(source_signal);
  ISC_noise = randn(size(source_signal));
  if isequal(distr, 'Chi2')
    ISC_noise = ISC_noise.^2;
  end
  ISC_noise = zscore(ISC_noise);
  source_signal = source_signal.*repmat(alphas, T, 1, N) + ISC_noise.*repmat(betas, T, 1, N);
  source_signal = zscore(source_signal);
  
%   diag(corr(source_signal(:, :, 1), source_signal(:, :, 2)))
  
  if samesignal
    % spatial distribution of the correlated activity,
    % also assumed to be the same for all subjects
    % if this is not the case (e.g. due to anatomical differences,
    % CCA is more appropriate than CorrCA
    Asignal = randpsd(D, min(K, D));
    for in = 1:N
      sensor_signal(:, :, in) = source_signal(:, :, in)*Asignal;
    end
  else
    % different signal correlation structure in each subject, 
    % representing, e.g., different subject anatomies
    for in = 1:N
      Asignal(:, :, in) = randpsd(D, min(K, D));
      sensor_signal(:, :, in) = source_signal(:, :, in)*Asignal(:, :, in);
    end
  end

end

if samenoise
  % sensor-space spatial correlation pattern of the noise
  % Crucially, also assumed to be the same here for all subjects
  % In a more realistic setting, this is different for all subjects, 
  % which would make CCA more appropriate than CorrCA
  Anoise = randpsd(D, min(T, D));
else
  % different noise spatial correlation structure in each subject
  for in = 1:N
    Anoise(:, :, in) = randpsd(D, min(T, D));
  end
end
  
data = zeros(T, D, N);
for in = 1:N
  % spatially correlated white or pink Gaussian noise in sensor space
  if pink
    source_noise(:, :, in) = mkpinknoise(T, min(T, D));
  else
    source_noise(:, :, in) = randn(T, min(T, D));
  end
    
  % make noise chi2 or Bernoulli distributed
  switch distr
    case 'Chi2'
      source_noise = source_noise.^2;
    case 'Bernoulli'
      source_noise = sign(source_noise);
  end
  
  if samenoise
    sensor_noise(:, :, in) = source_noise(:, :, in)*Anoise;
  else
    sensor_noise(:, :, in) = source_noise(:, :, in)*Anoise(:, :, in);
  end
  
  % spatially uncorrelated white sensor noise 
  meas_noise(:, :, in) = randn(T, min(T, D));
  
  switch distr
    case 'Chi2'
      meas_noise = meas_noise.^2;
%     case 'Bernoulli'
%       meas_noise = sign(meas_noise);
  end
  
end

sensor_signal = sensor_signal / norm(sensor_signal(:));
sensor_noise = sensor_noise / norm(sensor_noise(:));
meas_noise = meas_noise / norm(meas_noise(:));

noise = ssnr*meas_noise + (1-ssnr)*sensor_noise;
data = snr*sensor_signal + (1-snr)*noise;

if isequal(distr, 'Bernoulli')
  data = sign(data);
end

  
