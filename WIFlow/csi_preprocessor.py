

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision.utils import _log_api_usage_once

from WIFlow.torch_savgol import savgol_filter as torch_savgol_filter


class CSIPreprocessor(torch.nn.Module):
    """Preprocessing of Channel State Information(CSI)
    """

    def __init__(self, n_antenna, n_subcarrier,amplitude_max:float=3000):
        super().__init__()
        self.out_channels = n_antenna*2
        self.amplitude_max = amplitude_max
        _log_api_usage_once(self)

    def forward(self, csi:torch.Tensor)-> torch.Tensor:

        amp = csi.abs()/self.amplitude_max
        pha = csi.angle()
        features = torch.concat([amp,pha], axis=1)


        return features

    def __repr__(self) -> str:
        return self.__class__.__name__
class CSIPreprocessorAmp(CSIPreprocessor):
    def __init__(self, n_antenna, n_subcarrier,amplitude_max:float=3000):
        super().__init__(n_antenna, n_subcarrier, amplitude_max)
        self.out_channels = n_antenna

    def forward(self, csi):
        csi =  super().forward(csi)
        N, F, C, T = csi.shape # Batch, Features, Carrier, Time
        return csi[:, :F//2, :, :]

class CSIPreprocessor_dt(CSIPreprocessor):
    """Preprocessing of Channel State Information(CSI) derivitive in time dimension
    """
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier)
        self.out_channels = 2* self.out_channels
    def forward(self, csi:torch.Tensor):
        diff_dt = (F.pad(input=csi, pad=(1, 0, 0, 0), mode='reflect') - F.pad(input=csi, pad=(0, 1, 0, 0), mode='reflect'))[::,::,::,:-1]
        features = torch.concat([csi,diff_dt], axis=1)
        features =  super().forward(features)

        return features
class CSIPreprocessor_dc(CSIPreprocessor):
    """Preprocessing of Channel State Information(CSI) derivitive in (sub)carrier dimension
    """
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier)
        self.out_channels = self.out_channels*2
    def forward(self, csi):
        diff_dc= (F.pad(input=csi, pad=(0, 0, 1, 0), mode='reflect') - F.pad(input=csi, pad=(0, 0, 0, 1), mode='reflect'))[::,::,:-1,::]
        features = torch.concat([csi,diff_dc], axis=1)
        features = super().forward(features)

        return features

class __CSIPreprocessor_da(CSIPreprocessor):
    """Preprocessing of Channel State Information(CSI) derivitive in antenna dimension
    CAUTION: multiple devices need special consideration
    """
    def __init__(self, n_antenna, n_subcarrier, antenna_per_device:int=4):
        super().__init__(n_antenna, n_subcarrier)
        self.antenna_per_device = antenna_per_device
        self.out_channels = self.out_channels*2

    def forward(self, csi:torch.Tensor):
        result = []

        for device in csi.split(self.antenna_per_device,dim=1):
            diff_da = torch.subtract(device[::,:-1,::,::], device[::,1:,::,::])
            diff_da_overlapp = torch.subtract(device[::,-1:,::,::], device[::,0:1,::,::])
            result.append(torch.concat([device,diff_da,diff_da_overlapp], axis=1))

        features= torch.concat(result,dim=1)
        features = super().forward(features)

        return features


class CSIPreprocessor_da_seemo(__CSIPreprocessor_da):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier,antenna_per_device=4)
class CSIPreprocessor_da_mmfi(__CSIPreprocessor_da):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier,antenna_per_device=3)


class CSIPreprocessor_dtdc(CSIPreprocessor_dt,CSIPreprocessor_dc):
    pass
class CSIPreprocessor_dtdcda_seemo(CSIPreprocessor_da_seemo, CSIPreprocessor_dtdc):
    pass
class CSIPreprocessor_dtdcda_mmfi(CSIPreprocessor_da_mmfi, CSIPreprocessor_dtdc):
    pass

class TraiableCSIPreprocessor(CSIPreprocessor):
    def __init__(self,n_antenna,n_subcarrier,amplitude_max:float=3000):
        super().__init__(n_antenna, n_subcarrier)
        self.out_channels = 4* n_antenna # raw_pha,raw_amp, normalized_amp
        self.amplitude_normalization = Parameter(torch.zeros((1,n_antenna,n_subcarrier,1)),requires_grad=False)
        self.n_datapoints = Parameter(torch.zeros((1,)),requires_grad=False)
        self.amplitude_max = amplitude_max

    def forward(self, csi:torch.Tensor):
        N, A, C, T = csi.shape # Batch, Antenna, Carrier, Time

        amp_raw = csi.abs()/self.amplitude_max
        if self.training:
            if not self.n_datapoints:
                self.amplitude_normalization = Parameter(
                    self.amplitude_normalization +amp_raw.mean((0,3),keepdim=True),requires_grad=False)
            else:
                self.amplitude_normalization = Parameter(
                    self.amplitude_normalization/self.n_datapoints +amp_raw.mean((0,3),keepdim=True),requires_grad=False)
            self.n_datapoints += N*T
        pha_raw = csi.angle()
        pha_diff = (F.pad(input=pha_raw, pad=(0, 0, 1, 0), mode='reflect') - F.pad(input=pha_raw, pad=(0, 0, 0, 1), mode='reflect'))[::,::,:-1,::]

        features = torch.concat([amp_raw,pha_raw,amp_raw-self.amplitude_normalization, pha_diff], axis=1)
        return features



class TraiableNormCSIPreprocessorOnly(TraiableCSIPreprocessor):
    def __init__(self, n_antenna, n_subcarrier, amplitude_max = 3000, *args, **kwargs):
        super().__init__(n_antenna, n_subcarrier, amplitude_max, *args, **kwargs)
        self.out_channels = 2* n_antenna
        self.n_antenna = n_antenna
    def forward(self, csi):

        return super().forward(csi)[:,self.n_antenna*2:,:,:]





class FourierInversePreprocessorSub(CSIPreprocessor):
    def __init__(self, n_antenna, n_subcarrier, amplitude_max = 3000, dim=2, *args, **kwargs):
        self.dim=dim
        super().__init__(n_antenna, n_subcarrier, amplitude_max, *args, **kwargs)
    def forward(self, csi):
        fourier = torch.fft.ifft(csi, dim=2)
        splitted = torch.view_as_real(fourier)
        norm = torch.linalg.norm(splitted, dim=(3), keepdim=True) + 1e-8
        splitted_norm = splitted / norm
        real = splitted_norm[::,::,::,:,0]
        imag = splitted_norm[::,::,::,:,1]
        features = torch.concat([real,imag], axis=1)
        return features

class FourierInversePreprocessorTime(CSIPreprocessor):
    def __init__(self, n_antenna, n_subcarrier, amplitude_max = 3000, *args, **kwargs):
        super().__init__(n_antenna, n_subcarrier, amplitude_max, *args, **kwargs)
    def forward(self, csi):
        fourier = torch.fft.ifft(csi, dim=3)
        splitted = torch.view_as_real(fourier)
        norm = torch.linalg.norm(splitted, dim=(3), keepdim=True) + 1e-8
        splitted_norm = splitted / norm
        real = splitted_norm[::,::,::,:,0]
        imag = splitted_norm[::,::,::,:,1]
        features = torch.concat([real,imag], axis=1)
        return features








class TraiableCSIPreprocessordAdCdT(TraiableCSIPreprocessor,__CSIPreprocessor_da,CSIPreprocessor_dc,CSIPreprocessor_dt):
    pass


class FourierInversePreprocessordAdCdT(TraiableCSIPreprocessordAdCdT):
    def __init__(self, n_antenna, n_subcarrier, amplitude_max = 3000, dim=2, *args, **kwargs):
        self.dim=dim
        super().__init__(n_antenna, n_subcarrier, amplitude_max, *args, **kwargs)
        self.out_channels = 2* self.out_channels
    def forward(self, csi:torch.Tensor):
        csi = super().forward(csi)
        splitted = torch.concat([csi.abs()[..., None], csi.angle()[..., None]], dim=-1)
        norm = torch.linalg.norm(splitted, dim=(3), keepdim=True) + 1e-8
        splitted_norm = splitted / norm
        real = splitted_norm[::,::,::,:,0]
        imag = splitted_norm[::,::,::,:,1]
        features = torch.concat([real,imag], axis=1)
        return features
class FourierInversePreprocessord(CSIPreprocessor):
    def __init__(self, n_antenna, n_subcarrier, amplitude_max = 3000, dim=2, *args, **kwargs):
        self.dim=dim
        super().__init__(n_antenna, n_subcarrier, amplitude_max, *args, **kwargs)
        self.out_channels = 2* n_antenna
    def forward(self, csi:torch.Tensor):
        csi = super().forward(csi)
        splitted = torch.concat([csi.abs()[..., None], csi.angle()[..., None]], dim=-1)
        norm = torch.linalg.norm(splitted, dim=(3), keepdim=True) + 1e-8
        splitted_norm = splitted / norm
        real = splitted_norm[::,::,::,:,0]
        imag = splitted_norm[::,::,::,:,1]
        features = torch.concat([real,imag], axis=1)
        return features

class MeanNormalizationPhaConjPreprocessor(CSIPreprocessor):
    """Preprocessing of Channel State Information(CSI) with mean normalization of amplitude and phase conj from Fabian Portner
    """
    def __init__(self, n_antenna, n_subcarrier, devices=4):
        super().__init__(n_antenna, n_subcarrier)
        self.n_antenna_per_device = n_antenna // devices
        self.n_devices = devices
        self.out_channels = ((n_antenna//devices)-1)*devices*2 #-1 because of the lost ref antenna of comp. conj.


    def csi_phase_conj(self, csi:torch.Tensor)-> torch.Tensor:
        result = []
        for device_idx in range(csi.shape[1]//self.n_devices):
            csi_device = csi[:,device_idx*self.n_antenna_per_device:(device_idx+1)*self.n_antenna_per_device]
            reference_antenna = csi_device[:,3:] # last is reference
            normalized = csi_device[:,:3] * torch.conj(reference_antenna)
            result.append(normalized)
        relative_streams:torch.Tensor = torch.concat(result, axis=1)
        pha = relative_streams.angle() + torch.pi
        amp = relative_streams.abs()
        result = torch.concat([amp,pha], axis=1)
        return result

    def csi_amp_mean_norm(self, csi:torch.Tensor)-> torch.Tensor:
        csi_amp = csi.abs()
        csi = csi / (torch.mean(csi_amp, dim=2, keepdims=True)+ 1e-8)
        return csi
    def forward(self, csi:torch.Tensor):
        csi = self.csi_amp_mean_norm(csi)
        result = self.csi_phase_conj(csi)
        assert result.shape[1] == self.out_channels, f"Output channels {result.shape[1]} do not match expected {self.out_channels}"
        return result

class SingleDeviceMeanNormalizationPhaConjPreprocessor(MeanNormalizationPhaConjPreprocessor):
    def __init__(self, n_antenna:int, n_subcarrier:int, device_idx:int):
        super().__init__(n_antenna, n_subcarrier)
        self.device_idx = device_idx
        self.out_channels = self.out_channels//4
    def forward(self, csi:torch.Tensor):
        csi = csi[:, self.device_idx*self.n_antenna_per_device:(self.device_idx+1)*self.n_antenna_per_device,:,:]
        return super().forward(csi)

class SingleDeviceMeanNormalizationPhaConj0Preprocessor(SingleDeviceMeanNormalizationPhaConjPreprocessor):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, 0)
class SingleDeviceMeanNormalizationPhaConj1Preprocessor(SingleDeviceMeanNormalizationPhaConjPreprocessor):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, 1)
class SingleDeviceMeanNormalizationPhaConj2Preprocessor(SingleDeviceMeanNormalizationPhaConjPreprocessor):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, 2)
class SingleDeviceMeanNormalizationPhaConj3Preprocessor(SingleDeviceMeanNormalizationPhaConjPreprocessor):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, 3)



class WaveletPreprocessor(CSIPreprocessor):
    """Preprocessing of Channel State Information(CSI) with Wavelet Transform inspired through https://github.com/aiotgroup/Person-in-WiFi-3D-repo
    """
    def __init__(self, n_antenna, n_subcarrier):
        import pywt
        super().__init__(n_antenna, n_subcarrier)
        self.n_antenna_per_device = 4
        self.n_antenna = n_antenna
        self.n_subcarrier = n_subcarrier
        self.wavelet = pywt.Wavelet('dB11')
        self.out_channels = n_antenna *2 # real and imag part

    def forward(self, csi:torch.Tensor):
        device = csi.device
        csi_amp = self.amplitude(csi.to("cpu"))

        csi_phase = self.phase_deno(csi.to("cpu"))
        result = torch.concat([csi_amp, csi_phase], axis=1)
        return result.to(device)

    def amplitude(self, csi):
        import pywt
        list = pywt.wavedec(abs(csi), self.wavelet,'sym')
        csi_amp = pywt.waverec(list, self.wavelet)
        return torch.Tensor(csi_amp)

    def phase_deno(self, csi):
        csi_ = csi.numpy()
        B,A,C,T = csi.shape
        total_result = []
        for b in range(B):
            sample_result = []
            for index in range(0,A,self.n_antenna_per_device):
                device_csi = torch.tensor(self.CSI_sanitization(csi_[b,:,:][index:index+self.n_antenna_per_device]),dtype=torch.complex64)
                sample_result.append(device_csi)
            total_result.append(torch.concat(sample_result)[None])
        total_result = torch.concat(total_result)
        return torch.angle(total_result)
    def CSI_sanitization(self, csi_rx:np.ndarray)->np.ndarray:
        one_csi = csi_rx[0,:,:]
        two_csi = csi_rx[1,:,:]
        three_csi = csi_rx[2,:,:]
        four_csi = csi_rx[3,:,:]
        pi = np.pi
        T = csi_rx.shape[-1]  # total number of packets
        AN = csi_rx.shape[0]
        fi = 312.5 * 2  # subcarrier spacing 312.5 * 2 TODO this way be wrong
        csi_phase = np.zeros((AN, self.n_subcarrier, T))
        for t in range(T):  # iterate over CSI packets in time, each antenna has 30 subcarriers
            csi_phase[0, :, t] = np.unwrap(np.angle(one_csi[:, t]))
            csi_phase[1, :, t] = np.unwrap(csi_phase[0, :, t] + np.angle(two_csi[:, t] * np.conj(one_csi[:, t])))
            csi_phase[2, :, t] = np.unwrap(csi_phase[1, :, t] + np.angle(three_csi[:, t] * np.conj(two_csi[:, t])))
            csi_phase[3, :, t] = np.unwrap(csi_phase[2, :, t] + np.angle(four_csi[:, t] * np.conj(three_csi[:, t])))
            ai = np.tile(2 * pi * fi * np.array(range(self.n_subcarrier)), AN)
            bi = np.ones(AN * self.n_subcarrier)
            ci = np.concatenate((csi_phase[0, :, t], csi_phase[1, :, t], csi_phase[2, :, t],csi_phase[3, :, t]))
            A = np.dot(ai, ai)
            B = np.dot(ai, bi)
            C = np.dot(bi, bi)
            D = np.dot(ai, ci)
            E = np.dot(bi, ci)
            rho_opt = (B * E - C * D) / (A * C - B ** 2)
            beta_opt = (B * D - A * E) / (A * C - B ** 2)
            temp = np.tile(np.array(range(self.n_subcarrier)), AN).reshape(AN, self.n_subcarrier)
            csi_phase[:, :, t] = csi_phase[:, :, t] + 2 * pi * fi * temp * rho_opt + beta_opt
        antennaPair_One = abs(one_csi) * np.exp(1j * csi_phase[0, :, :])
        antennaPair_Two = abs(two_csi) * np.exp(1j * csi_phase[1, :, :])
        antennaPair_Three = abs(three_csi) * np.exp(1j * csi_phase[2, :, :])
        antennaPair_Four = abs(four_csi) * np.exp(1j * csi_phase[3, :, :])
        antennaPair = np.concatenate((np.expand_dims(antennaPair_One,axis=0),
                                      np.expand_dims(antennaPair_Two,axis=0),
                                      np.expand_dims(antennaPair_Three,axis=0),
                                      np.expand_dims(antennaPair_Four,axis=0),))
        return antennaPair




class CSIQuotientPreprocessor(CSIPreprocessor):
    """CSI Qoutient preprocessor based on Witraj https://github.com/Soccerene/WiTraj/tree/main"""
    def __init__(self, n_antenna, n_subcarrier, devices=4):
        super().__init__(n_antenna, n_subcarrier)
        self.n_antenna_per_device = 4
        self.n_antenna = n_antenna
        self.n_subcarrier = n_subcarrier
        self.out_channels = ((n_antenna//devices)-1)*devices*2
    def forward(self, csi:torch.Tensor):
        csiq = self.csi_quotient(csi)
        amp = csiq.abs()
        amp = amp.clip(0,amp.quantile(0.99))
        return torch.concat([amp,csiq.angle()], dim=1)
    def csi_quotient(self, csi:torch.Tensor):
        B,A,C,T = csi.shape
        total_result = []
        for index in range(0,A,self.n_antenna_per_device):
            quotient_antenna = csi[:,index+self.n_antenna_per_device-1:index+self.n_antenna_per_device,:,:] + 1e-9j + 1e-9
            csi_q = csi[:,index:index+self.n_antenna_per_device-1,:,:]/(quotient_antenna)

            total_result.append(csi_q)
        total_result = torch.concat(total_result,dim=1)
        return total_result

class SingleDeviceCSIQuotientPreprocessor(CSIQuotientPreprocessor):
    def __init__(self, n_antenna:int, n_subcarrier:int, device_idx:int):
        super().__init__(n_antenna, n_subcarrier)
        self.device_idx = device_idx
        self.out_channels = self.out_channels//4
    def forward(self, csi:torch.Tensor):
        csi = csi[:, self.device_idx*self.n_antenna_per_device:(self.device_idx+1)*self.n_antenna_per_device,:,:]
        return super().forward(csi)

class SingleDeviceQuotient0Preprocessor(SingleDeviceCSIQuotientPreprocessor):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, 0)
class SingleDeviceQuotient1Preprocessor(SingleDeviceCSIQuotientPreprocessor):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, 1)
class SingleDeviceQuotient2Preprocessor(SingleDeviceCSIQuotientPreprocessor):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, 2)
class SingleDeviceQuotient3Preprocessor(SingleDeviceCSIQuotientPreprocessor):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, 3)



class CSIQuotientAntennaReductionPreprocessor(CSIQuotientPreprocessor):
    """CSI Antenna reduction to proof claim of Witraj"""
    def __init__(self, n_antenna, n_subcarrier, orig_devices:int=4, remove_device:int=0):
        super().__init__(n_antenna, n_subcarrier)
        self.antenna_per_device = n_antenna // orig_devices
        self.preserved_antennas = self.antenna_per_device * (orig_devices - remove_device)
        self.out_channels = ((self.preserved_antennas//(orig_devices - remove_device))-1)*(orig_devices-remove_device)*2 # because one antenna per device is used as reference for quotient, so -1
    def forward(self, csi:torch.Tensor):
        csi = csi[:, :self.preserved_antennas, :, :]
        return super().forward(csi)

class CSIQuotientAntennaReduction0Preprocessor(CSIQuotientAntennaReductionPreprocessor):
    """CSI Antenna reduction to proof claim of Witraj"""
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, orig_devices=4, remove_device=0)
class CSIQuotientAntennaReduction1Preprocessor(CSIQuotientAntennaReductionPreprocessor):
    """CSI Antenna reduction to proof claim of Witraj"""
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, orig_devices=4, remove_device=1)
class CSIQuotientAntennaReduction2Preprocessor(CSIQuotientAntennaReductionPreprocessor):
    """CSI Antenna reduction to proof claim of Witraj"""
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, orig_devices=4, remove_device=2)
class CSIQuotientAntennaReduction3Preprocessor(CSIQuotientAntennaReductionPreprocessor):
    """CSI Antenna reduction to proof claim of Witraj"""
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, orig_devices=4, remove_device=3)


class SingleAntennaCSIQuotientPreprocessor(CSIQuotientPreprocessor):
    def __init__(self, n_antenna:int, n_subcarrier:int, device_idx:int=0, antenna_idx:int=0):
        super().__init__(n_antenna, n_subcarrier)
        self.device_idx = device_idx
        self.antenna_idx = antenna_idx
        self.n_antenna_per_device = 2
        self.out_channels = 2 # phase and amplitude
    def forward(self, csi:torch.Tensor):
        csi = csi[:, self.device_idx+self.antenna_idx:(self.device_idx+2)+self.antenna_idx,:,:] # still need two for quotient, so device_idx+2,
        return super().forward(csi)



class CSIQuotientSavGolPreprocessor(CSIQuotientPreprocessor):
    """CSI Qoutient preprocessor based on Witraj https://github.com/Soccerene/WiTraj/tree/main"""

    def forward(self, csi:torch.Tensor):
        device = csi.device
        csiq = self.csi_quotient(csi)
        csiq = self.apply_savgol_filter(csiq)
        csiq = csiq.to(device)
        amp = csiq.abs()
        amp = amp.clip(0,amp.quantile(0.99))
        return torch.concat([amp,csiq.angle()], dim=1)

    def apply_savgol_filter(self, csi:torch.Tensor)->torch.Tensor:
        from scipy.signal import savgol_filter as scipy_savgol_filter
        B,A,C,T = csi.shape
        csi_np = csi

        csi_flat = csi_np.reshape(-1, T)

        def apply_filter_1d(time_series):
            return scipy_savgol_filter(time_series.real, 51, 2) + 1j * scipy_savgol_filter(time_series.imag, 51, 2)

        filtered_flat = np.array([apply_filter_1d(ts) for ts in csi_flat])
        filtered = filtered_flat.reshape(B, A, C, T)
        return torch.tensor(filtered, dtype=torch.complex64)




class CSIQuotientSavGolTorchPreprocessor(CSIQuotientSavGolPreprocessor):
    """CSI Qoutient preprocessor based on Witraj https://github.com/Soccerene/WiTraj/tree/main"""
    def __init__(self, n_antenna, n_subcarrier, devices=4, conv_mode="interp"):
        super().__init__(n_antenna, n_subcarrier, devices)
        self.conv_mode = conv_mode
    def forward(self, csi:torch.Tensor):
        csiq = self.csi_quotient(csi)
        csiq = self.apply_savgol_filter(csiq)
        amp = csiq.abs()
        amp = amp.clip(0,amp.quantile(0.99))
        return torch.concat([amp,csiq.angle()], dim=1)

    def apply_savgol_filter(self, csi:torch.Tensor)->torch.Tensor:
        B, A, C, T = csi.shape
        csi_flat = csi.reshape(-1, T)

        def apply_filter_1d(time_series):
            return torch_savgol_filter(time_series.real, 51, 2, mode=self.conv_mode) + 1j * torch_savgol_filter(time_series.imag, 51, 2, mode=self.conv_mode)

        filtered = torch.vmap(apply_filter_1d)(csi_flat)
        filtered = filtered.reshape(B, A, C, T)
        return torch.tensor(filtered, dtype=torch.complex64)



class CSISavGolTorchPreprocessor(CSIQuotientSavGolTorchPreprocessor):
    def __init__(self, n_antenna, n_subcarrier, devices=4, conv_mode="interp"):
        super().__init__(n_antenna, n_subcarrier, devices)
        self.conv_mode = conv_mode
        self.out_channels = n_antenna*2
    def forward(self, csi:torch.Tensor):
        csi = self.apply_savgol_filter(csi)
        amp = csi.abs()
        amp = amp.clip(0,amp.quantile(0.99))
        return torch.concat([amp,csi.angle()], dim=1)





class CSIAntennaReductionPreprocessor(CSIPreprocessor):
    """CSI Antenna reduction to proof claim of Witraj"""
    def __init__(self, n_antenna, n_subcarrier, orig_devices:int=4, remove_device:int=0):
        super().__init__(n_antenna, n_subcarrier)
        self.antenna_per_device = n_antenna // orig_devices
        self.preserved_antennas = self.antenna_per_device * (orig_devices - remove_device)
        self.out_channels = self.preserved_antennas*2
    def forward(self, csi:torch.Tensor):
        csi = csi[:, :self.preserved_antennas, :, :]
        return super().forward(csi)

class CSIAntennaReduction0Preprocessor(CSIAntennaReductionPreprocessor):
    """CSI Antenna reduction to proof claim of Witraj"""
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, orig_devices=4, remove_device=0)
class CSIAntennaReduction1Preprocessor(CSIAntennaReductionPreprocessor):
    """CSI Antenna reduction to proof claim of Witraj"""
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, orig_devices=4, remove_device=1)
class CSIAntennaReduction2Preprocessor(CSIAntennaReductionPreprocessor):
    """CSI Antenna reduction to proof claim of Witraj"""
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, orig_devices=4, remove_device=2)
class CSIAntennaReduction3Preprocessor(CSIAntennaReductionPreprocessor):
    """CSI Antenna reduction to proof claim of Witraj"""
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, orig_devices=4, remove_device=3)


class CSISingleDevicePreprocessor(CSIPreprocessor):
    """CSI Preprocessor for single device to proof claim of Witraj"""
    def __init__(self, n_antenna:int, n_subcarrier:int, device_idx:int):
        super().__init__(n_antenna, n_subcarrier)
        self.antenna_per_device = n_antenna // 4
        self.device_idx = device_idx
        self.out_channels = self.antenna_per_device*2
    def forward(self, csi:torch.Tensor):
        csi = csi[:, self.device_idx*self.antenna_per_device:(self.device_idx+1)*self.antenna_per_device,:,:]
        return super().forward(csi)

class CSISingleDevice0Preprocessor(CSISingleDevicePreprocessor):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, device_idx=0)
class CSISingleDevice1Preprocessor(CSISingleDevicePreprocessor):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, device_idx=1)
class CSISingleDevice2Preprocessor(CSISingleDevicePreprocessor):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, device_idx=2)
class CSISingleDevice3Preprocessor(CSISingleDevicePreprocessor):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, device_idx=3)



class PCAPreprocessor(CSIPreprocessor):
    def __init__(self, n_antenna, n_subcarrier, saved_components_path="pca_components.pt", preserved_components:int=None):
        super().__init__( n_antenna, n_subcarrier,)

        self.components = torch.load(saved_components_path)
        if preserved_components:
            self.components = self.components[:,:preserved_components]
        self.output_dim = n_antenna * 2

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        N, A, C, T = csi.shape
        self.components = self.components.to(csi.device)
        tensor = super().forward(csi)
        tensor_flat = tensor.permute(0,3,1,2).flatten(start_dim=2).flatten(0,1)
        transformed = tensor_flat @ self.components
        denoised_flat = transformed @ self.components.T

        denoised = denoised_flat.reshape(N, T, 2*A, C).permute(0, 2, 3, 1)
        return denoised

class PCAPreprocessorPreserved3000(PCAPreprocessor):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, preserved_components=3000)
class PCAPreprocessorPreserved2000(PCAPreprocessor):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, preserved_components=2000)
class PCAPreprocessorPreserved1500(PCAPreprocessor):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, preserved_components=1500)
class PCAPreprocessorPreserved750(PCAPreprocessor):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, preserved_components=750)
class PCAPreprocessorPreserved375(PCAPreprocessor):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, preserved_components=375)
class PCAPreprocessorPreserved150(PCAPreprocessor):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, preserved_components=150)

class PCAPreprocessorPreserved20(PCAPreprocessor):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, preserved_components=20)

class PCA150DeviceReduction(PCAPreprocessorPreserved150):
    def __init__(self, n_antenna, n_subcarrier,orig_devices:int=4, remove_device:int=0):
        super().__init__(n_antenna, n_subcarrier)
        self.antenna_per_device = n_antenna // orig_devices
        self.preserved_antennas = self.antenna_per_device * (orig_devices - remove_device)
        self.out_channels = self.preserved_antennas*2

        half_dim = self.components.shape[0] // 2
        amp_components = self.components[:half_dim, :]
        phase_components = self.components[half_dim:, :]

        preserved_dim = self.preserved_antennas * n_subcarrier

        amp_components = amp_components[:preserved_dim, :]
        phase_components = phase_components[:preserved_dim, :]
        self.components = torch.cat([amp_components, phase_components], dim=0)
    def forward(self, csi:torch.Tensor):
        csi = csi[:, :self.preserved_antennas, :, :]
        return super().forward(csi)
class PCA150DeviceReduction0(PCA150DeviceReduction):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, remove_device=0)
class PCA150DeviceReduction1(PCA150DeviceReduction):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, remove_device=1)
class PCA150DeviceReduction2(PCA150DeviceReduction):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, remove_device=2)
class PCA150DeviceReduction3(PCA150DeviceReduction):
    def __init__(self, n_antenna, n_subcarrier):
        super().__init__(n_antenna, n_subcarrier, remove_device=3)
