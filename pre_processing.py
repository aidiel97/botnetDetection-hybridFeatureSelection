import socket, struct
import numpy as np

stateDict = {'': 1, 'FSR_SA': 30, '_FSA': 296, 'FSRPA_FSA': 77, 'SPA_SA': 31, 'FSA_SRA': 1181, 'FPA_R': 46, 'SPAC_SPA': 37, 'FPAC_FPA': 2, '_R': 1, 'FPA_FPA': 784, 'FPA_FA': 66, '_FSRPA': 1, 'URFIL': 431, 'FRPA_PA': 5, '_RA': 2, 'SA_A': 2, 'SA_RA': 125, 'FA_FPA': 17, 'FA_RA': 14, 'PA_FPA': 48, 'URHPRO': 380, 'FSRPA_SRA': 8, 'R_':541, 'DCE': 5, 'SA_R': 1674, 'SA_': 4295, 'RPA_FSPA': 4, 'FA_A': 17, 'FSPA_FSPAC': 7, 'RA_': 2230, 'FSRPA_SA': 255, 'NNS': 47, 'SRPA_FSPAC': 1, 'RPA_FPA': 42, 'FRA_R': 10, 'FSPAC_FSPA': 86, 'RPA_R': 3, '_FPA': 5, 'SREC_SA': 1, 'URN': 339, 'URO': 6, 'URH': 3593, 'MRQ': 4, 'SR_FSA': 1, 'SPA_SRPAC': 1, 'URP': 23598, 'RPA_A': 1, 'FRA_': 351, 'FSPA_SRA': 91, 'FSA_FSA': 26138, 'PA_': 149, 'FSRA_FSPA': 798, 'FSPAC_FSA': 11, 'SRPA_SRPA': 176, 'SA_SA': 33, 'FSPAC_SPA': 1, 'SRA_RA': 78, 'RPAC_PA': 1, 'FRPA_R': 1, 'SPA_SPA': 2989, 'PA_RA': 3, 'SPA_SRPA': 4185, 'RA_FA': 8, 'FSPAC_SRPA': 1, 'SPA_FSA': 1, 'FPA_FSRPA': 3, 'SRPA_FSA': 379, 'FPA_FRA': 7, 'S_SRA': 81, 'FSA_SA': 6, 'State': 1, 'SRA_SRA': 38, 'S_FA': 2, 'FSRPAC_SPA': 7, 'SRPA_FSPA': 35460, 'FPA_A': 1, 'FSA_FPA': 3, 'FRPA_RA': 1, 'FSAU_SA': 1, 'FSPA_FSRPA': 10560, 'SA_FSA': 358, 'FA_FRA': 8, 'FSRPA_SPA': 2807, 'FSRPA_FSRA': 32, 'FRA_FPA': 6, 'FSRA_FSRA': 3, 'SPAC_FSRPA': 1, 'FS_': 40, 'FSPA_FSRA': 798, 'FSAU_FSA': 13, 'A_R': 36, 'FSRPAE_FSPA': 1, 'SA_FSRA': 4, 'PA_PAC': 3, 'FSA_FSRA': 279, 'A_A': 68, 'REQ': 892, 'FA_R': 124, 'FSRPA_SRPA': 97, 'FSPAC_FSRA':20, 'FRPA_RPA': 7, 'FSRA_SPA': 8, 'INT': 85813, 'FRPA_FRPA': 6, 'SRPAC_FSPA': 4, 'SPA_SRA': 808, 'SA_SRPA': 1, 'SPA_FSPA': 2118, 'FSRAU_FSA': 2, 'RPA_PA': 171,'_SPA': 268, 'A_PA': 47, 'SPA_FSRA': 416, 'FSPA_FSRPAC': 2, 'PAC_PA': 5, 'SRPA_SPA': 9646, 'SRPA_FSRA': 13, 'FPA_FRPA': 49, 'SRA_SPA': 10, 'SA_SRA': 838, 'PA_PA': 5979, 'FPA_RPA': 27, 'SR_RA': 10, 'RED': 4579, 'CON': 2190507, 'FSRPA_FSPA':13547, 'FSPA_FPA': 4, 'FAU_R': 2, 'ECO': 2877, 'FRPA_FPA': 72, 'FSAU_SRA': 1, 'FRA_FA': 8, 'FSPA_FSPA': 216341, 'SEC_RA': 19, 'ECR': 3316, 'SPAC_FSPA': 12, 'SR_A': 34, 'SEC_': 5, 'FSAU_FSRA': 3, 'FSRA_FSRPA': 11, 'SRC': 13, 'A_RPA': 1, 'FRA_PA': 3, 'A_RPE': 1, 'RPA_FRPA': 20, '_SRA': 74, 'SRA_FSPA': 293, 'FPA_': 118, 'FSRPAC_FSRPA': 2, '_FA': 1, 'DNP': 1, 'FSRPA_FSRPA': 379, 'FSRA_SRA': 14, '_FRPA': 1, 'SR_': 59, 'FSPA_SPA': 517, 'FRPA_FSPA': 1, 'PA_A': 159, 'PA_SRA': 1, 'FPA_RA': 5, 'S_': 68710, 'SA_FSRPA': 4, 'FSA_FSRPA': 1, 'SA_SPA': 4, 'RA_A': 5, '_SRPA': 9, 'S_FRA': 156, 'FA_FRPA': 1, 'PA_R': 72, 'FSRPAEC_FSPA': 1, '_PA': 7, 'RA_S': 1, 'SA_FR': 2, 'RA_FPA': 6, 'RPA_': 5, '_FSPA': 2395, 'FSA_FSPA': 230, 'UNK': 2, 'A_RA': 9, 'FRPA_': 6, 'URF': 10, 'FS_SA': 97, 'SPAC_SRPA': 8, 'S_RPA': 32, 'SRPA_SRA': 69, 'SA_RPA': 30, 'PA_FRA': 4, 'FSRA_SA': 49, 'FSRA_FSA': 206, 'PAC_RPA': 1, 'SRA_': 18, 'FA_': 451, 'S_SA': 6917, 'FSPA_SRPA': 427, 'TXD': 542,'SRA_SA': 1514, 'FSPA_FA': 1, 'FPA_FSPA': 10, 'RA_PA': 3, 'SRA_FSA': 709, 'SRPA_SPAC': 3, 'FSPAC_FSRPA': 10, 'A_': 191, 'URNPRO': 2, 'PA_RPA': 81, 'FSPAC_SRA':1, 'SRPA_FSRPA': 3054, 'SPA_': 1, 'FA_FA': 259, 'FSPA_SA': 75, 'SR_SRA': 1, 'FSA_': 2, 'SRPA_SA': 406, 'SR_SA': 3119, 'FRPA_FA': 1, 'PA_FRPA': 13, 'S_R': 34, 'FSPAEC_FSPAE': 3, 'S_RA': 61105, 'FSPA_FSA': 5326, '_SA': 20, 'SA_FSPA': 15, 'SRPAC_SPA': 8, 'FPA_PA': 19, 'FSRPAE_FSA': 1, 'S_A': 1, 'RPA_RPA': 3, 'NRS': 6, 'RSP': 115, 'SPA_FSRPA': 1144, 'FSRPAC_FSPA': 139}
protoDict = {'arp': 5, 'unas': 13, 'udp': 1, 'rtcp': 7, 'pim': 3, 'udt': 11, 'esp': 12, 'tcp' : 0, 'rarp': 14, 'ipv6-icmp': 9, 'rtp': 2, 'ipv6': 10, 'ipx/spx': 6, 'icmp': 4, 'igmp' : 8}
symbolMap = {
   '<': 1, '>': 2, '?': 3, '-': 4,
   '<-': 5, '->': 6, '<>': 7, '><': 8,
   '<?': 9, '>?': 10, '-?': 11,
   '<->': 12, '<?>': 13, '>-<': 14
}

def port(port_str):
    try:
        if isinstance(port_str, str) and port_str.startswith('0x'):
            return int(port_str, 16)
        
        return int(port_str)
    except ValueError:
        return None

def split_by_class(df, train_ratio=0.7):
    train_indices = []
    test_indices = []

    for cls in df['Label'].unique():
        cls_indices = df[df['Label'] == cls].index
        split_point = int(len(cls_indices) * train_ratio)
        train_indices.extend(cls_indices[:split_point])
        test_indices.extend(cls_indices[split_point:])

    return df.loc[train_indices], df.loc[test_indices]

def ipToInteger(ip):
  try:
    packedIP = socket.inet_aton(ip)
    return struct.unpack("!L", packedIP)[0]
  except OSError:
    return np.nan

def label(label):
	label = label.lower()
	if 'botnet' in label:
		return 1
	else:
		return 0
  
def main(pd, data):
	data['StartTime'] = pd.to_datetime(data['StartTime']).astype(np.int64) // 10**9
	data['SrcAddr'] = data['SrcAddr'].apply(ipToInteger).fillna(0).astype(int)
	data['DstAddr'] = data['DstAddr'].apply(ipToInteger).fillna(0).astype(int)
	data['Dur'] = data['Dur'].fillna(0).astype(float)
	data['Proto'] = data['Proto'].map(protoDict).fillna(0).astype(int)
	data['Dir'] = data['Dir'].map(symbolMap).fillna(0).astype(int)
	data['State'] = data['State'].map(stateDict).fillna(0).astype(int)
	data['Sport']= data['Sport'].apply(port).fillna(0)
	data['Dport']= data['Dport'].apply(port).fillna(0)
	data['sTos'] = pd.to_numeric(data['sTos'], errors='coerce').fillna(0)
	data['dTos'] = pd.to_numeric(data['dTos'], errors='coerce').fillna(0)
	data['TotPkts'] = pd.to_numeric(data['TotPkts'], errors='coerce').fillna(0).astype(int)
	data['TotBytes'] = pd.to_numeric(data['TotBytes'], errors='coerce').fillna(0).astype(int)
	data['SrcBytes'] = pd.to_numeric(data['SrcBytes'], errors='coerce').fillna(0).astype(int)
	
	data = data.astype({
	'StartTime': 'int64',
	'Dur': 'float64',
	'Proto': 'int32',
	'Sport': 'int32',
	'Dir': 'int32',
	'Dport': 'int32',
	'State': 'int32',
	'sTos': 'float64',
	'dTos': 'float64',
	'TotPkts': 'int32',
	'TotBytes': 'int32',
	'SrcBytes': 'int32',
	'Label': 'int32'
	})

	return data