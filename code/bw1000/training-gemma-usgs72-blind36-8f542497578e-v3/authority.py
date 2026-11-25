"""Fixed one-time no-score scope and fresh root-owned placement."""
import re
import time

from asset_witness import INDEPENDENT_SHA, OWNER_SHA, PUBLIC_TERMINAL_SHA
from contract import DISPATCH_REQUEST, LABEL, NODE, PARENT, PORT, REQUEST, WHOLE, need


def digest(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def verify_authority(value,source,epoch,publication):
    need(set(value)=={'request','dispatch_request','label','parent','node','port','source_manifest',
        'publication','issued_epoch','root_GPU_GO','same_business_placement_confirmed','no_gold_target_staged',
        'score_GO','reservation_SHA256','fresh_physical_SHA256','expected_compute_boot_id','physical_PCI',
        'asset_independent_SHA256','asset_owner_outcome_SHA256','asset_public_terminal_SHA256'},
        'exact root authority schema')
    need((value['request'],value['dispatch_request'],value['label'],value['parent'],value['node'],value['port'])
         ==(REQUEST,DISPATCH_REQUEST,LABEL,PARENT,NODE,PORT) and value['source_manifest']==source
         and value['publication']==publication,'exact owner/publication and new scope required')
    need(value['root_GPU_GO'] is value['same_business_placement_confirmed'] is value['no_gold_target_staged'] is True
         and value['score_GO'] is False and digest(value['reservation_SHA256']) and digest(value['fresh_physical_SHA256']),
         'one explicit root GPU approval, protected reservation, and no score required')
    need((value['asset_independent_SHA256'],value['asset_owner_outcome_SHA256'],value['asset_public_terminal_SHA256'])
         ==(INDEPENDENT_SHA,OWNER_SHA,PUBLIC_TERMINAL_SHA),'exact .3538 terminal/owner proof required')
    boot=value['expected_compute_boot_id']
    regex='[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}'
    pci=value['physical_PCI']
    need(isinstance(boot,str) and re.fullmatch(regex,boot) is not None
         and type(pci) is list and len(pci)==len(set(pci))==8
         and all(isinstance(item,str) and re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',item) for item in pci),
         'actual fresh same-node boot and eight unique PCI required')
    need(type(value['issued_epoch']) in (int,float) and type(epoch) in (int,float)
         and -3<=epoch-value['issued_epoch']<=90 and time.time()<epoch+WHOLE,'root placement authority freshness differs')


def verify_physical(value,witness):
    need(witness['boot_id']==value['expected_compute_boot_id']
         and set(witness['pci'])==set(value['physical_PCI']) and witness['kfd_holder_count']==0
         and witness['full_fd_inventory_performed'] is True,'fresh root-approved same boot/eight PCI gate differs')
