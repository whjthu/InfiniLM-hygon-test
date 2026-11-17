"""Strict T03/T04 native input-reference metadata; stdlib only, no model imports."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import stat

CONTRACT_KEYS = ['spec', 'resolved_template_type', 'target_modules', 'trainable_manifest', 'model_source_SHA256', 'native_checkpointing_lifecycle', 'capture_budget_conditions']
CONTRACTS = {'T03.sft': {'semantic_SHA256': '93539463446b138f0e685361dccf38f880e2d2a7f7ec4cc1a1ef333a1ff36385', 'original_binding_SHA256': '2ff3b19fc433f38b2b9053787f1b458e70ada2457e1d1092f5beab3ae769a076', 'reference_request_id': '031e04f24efa429595364e6f0da1e5b6', 'run_root': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1', 'source_receipts': {'0': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1/capture/rank-00/receipt.json', 'bytes': 4302, 'SHA256': '9922d23a0a29037df2e25a3059cbae1082c0021918c248d477c62c3498660c90'}, '1': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1/capture/rank-01/receipt.json', 'bytes': 4299, 'SHA256': '9baf5444fc9a31b9cf8546742db00cda75631c94bd1190605af27eb8425165ff'}, '2': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1/capture/rank-02/receipt.json', 'bytes': 4301, 'SHA256': '518da3273f76884048eca6efa958c05441b3c67fe26ef1dd8a1b9ce88ee8b22e'}, '3': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1/capture/rank-03/receipt.json', 'bytes': 4303, 'SHA256': '741ee896d30d0ac41b6f400372b7a34b7fd9454dd1904b27f098286d8df2fafc'}, '4': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1/capture/rank-04/receipt.json', 'bytes': 4299, 'SHA256': '644f684227f9c8450b5d2bcbe635e54a8fb493412e8111397da5a59688a1a291'}, '5': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1/capture/rank-05/receipt.json', 'bytes': 4300, 'SHA256': '037dad4b27f3231aaeedfee235fcc44c9c8ff836ead65b24ef37a23bc007cf44'}, '6': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1/capture/rank-06/receipt.json', 'bytes': 4300, 'SHA256': 'e313c525860ffd92a6407b2d27877b61acd304719325aed1a99335219a47eb67'}, '7': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1/capture/rank-07/receipt.json', 'bytes': 4302, 'SHA256': '3586e344b9d2d3a9f6034d3e449176a7b465d01ec160d2465c97f8dc66671959'}}, 'original_value_audit_SHA256': '9664da5de388ea4687a0e58493e75c72baed61ef96efc90444b2754b7ccdc060'}, 'T04.sft': {'semantic_SHA256': '5eeb3edb715168db3ca9a5989291caab00a4bec385fa1043dc11de8914a8baff', 'original_binding_SHA256': 'a21db6fb5d16796c5e1f2c43722e1c1cac5303782b732c1c1ba7d45f58bca823', 'reference_request_id': '640a2e0f2ae4440a9689912f2de413be', 'run_root': '/home/whj/nvidia-baseline/alignment-assets/hygon-llava-A100-first-updates-v1', 'source_receipts': {'0': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-llava-A100-first-updates-v1/capture/rank-00/receipt.json', 'bytes': 4308, 'SHA256': '097543943d52c630fd95bf010e160a43b32f851aa962d3ec767868b74cb23684'}, '1': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-llava-A100-first-updates-v1/capture/rank-01/receipt.json', 'bytes': 4308, 'SHA256': 'f37d050e8e1f7b2f67156808a902e8e24065361d1ea791f1b5a315d8161b6d60'}, '2': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-llava-A100-first-updates-v1/capture/rank-02/receipt.json', 'bytes': 4308, 'SHA256': '17714a48e7e5c8785c057ad038e5017c4ad42b8902a3340c4c30624d550b255f'}, '3': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-llava-A100-first-updates-v1/capture/rank-03/receipt.json', 'bytes': 4308, 'SHA256': '87351ad63efe2a50f8237e73ae1776f344c704f006c567e5d40022cc217787fb'}, '4': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-llava-A100-first-updates-v1/capture/rank-04/receipt.json', 'bytes': 4308, 'SHA256': '42f49dcbbd7ae2fb73388842386eb83d82120097a926fa7fb0bca96b48eea358'}, '5': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-llava-A100-first-updates-v1/capture/rank-05/receipt.json', 'bytes': 4308, 'SHA256': '99a4c67edab9ef993f56bafebed97e1792ed506f65bdd85edbef58cb6bdbc662'}, '6': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-llava-A100-first-updates-v1/capture/rank-06/receipt.json', 'bytes': 4308, 'SHA256': 'f74092ff73c2c3821cfd7f89957b9fafd79dd0380f72f92f72ff0f8059d4d780'}, '7': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-llava-A100-first-updates-v1/capture/rank-07/receipt.json', 'bytes': 4308, 'SHA256': '2dcf92b8ad2ec6b9d23c2e43e193a88b3b869dc5d97b54851cd056dea813cf4d'}}, 'original_value_audit_SHA256': 'e78b6ccea81130cf92af72cf7aa9497f65bb1c33cff71572eaebde7e86d4df4b'}}
FINGERPRINT_SOURCE_SHA256 = '816ef694537273ac04d8f7d40df5694a16e8a869797d2d2f65664d21e6a0ea49'
MAX_REFERENCE_BYTES = 128 * 1024
ORIGINAL_INITIAL_SOURCE_SHA256 = '117f69b17afc83017315095dafe8c9b7253b761b2314b2cc14675e8db1140c42'
ORIGINAL_INITIAL_PINS = {'T03.sft': [{'rank': 0, 'initial_file_bytes': 174821899, 'initial_file_SHA256': '2a0df4d5fdd4d793d47859f7e6ee9102cd3324199f90d0ed34760d3c7198612f', 'named_initial_digest': '086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137'}, {'rank': 1, 'initial_file_bytes': 174821899, 'initial_file_SHA256': '03d006cf4242089e78bcf926b288124a307d675737a812a803526d6501805d3d', 'named_initial_digest': '086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137'}, {'rank': 2, 'initial_file_bytes': 174821899, 'initial_file_SHA256': '0039a5be7bd17cb0a1e9e065d6bc2a944d52e3d2290299dd9ded138417b48d1b', 'named_initial_digest': '086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137'}, {'rank': 3, 'initial_file_bytes': 174821899, 'initial_file_SHA256': 'ba9d40577785c2ff0d9a7f9d8d055a4a789dad70e7c29f39039e6b897efee26a', 'named_initial_digest': '086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137'}, {'rank': 4, 'initial_file_bytes': 174821899, 'initial_file_SHA256': 'af7bb01e108349ba6451da6ca614ef0ed5b38ae85e92831d9bd72db19e07a1ae', 'named_initial_digest': '086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137'}, {'rank': 5, 'initial_file_bytes': 174821899, 'initial_file_SHA256': '492f58eb1fae2045d25d116466b4816c5510bc22ef8b7abd5a44496037ff478c', 'named_initial_digest': '086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137'}, {'rank': 6, 'initial_file_bytes': 174821899, 'initial_file_SHA256': '38f104f8964c8fbeda0bdd71d2868ed50a7330e65fb81e9c25e03e3b18d9ff94', 'named_initial_digest': '086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137'}, {'rank': 7, 'initial_file_bytes': 174821899, 'initial_file_SHA256': '65e67bf381aec8e6ef0ccf5616d4b78cd82ac193c1aa26ed56e1445697961f00', 'named_initial_digest': '086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137'}], 'T04.sft': [{'rank': 0, 'initial_file_bytes': 174821899, 'initial_file_SHA256': '2a0df4d5fdd4d793d47859f7e6ee9102cd3324199f90d0ed34760d3c7198612f', 'named_initial_digest': '086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137'}, {'rank': 1, 'initial_file_bytes': 174821899, 'initial_file_SHA256': '03d006cf4242089e78bcf926b288124a307d675737a812a803526d6501805d3d', 'named_initial_digest': '086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137'}, {'rank': 2, 'initial_file_bytes': 174821899, 'initial_file_SHA256': '0039a5be7bd17cb0a1e9e065d6bc2a944d52e3d2290299dd9ded138417b48d1b', 'named_initial_digest': '086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137'}, {'rank': 3, 'initial_file_bytes': 174821899, 'initial_file_SHA256': 'ba9d40577785c2ff0d9a7f9d8d055a4a789dad70e7c29f39039e6b897efee26a', 'named_initial_digest': '086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137'}, {'rank': 4, 'initial_file_bytes': 174821899, 'initial_file_SHA256': 'af7bb01e108349ba6451da6ca614ef0ed5b38ae85e92831d9bd72db19e07a1ae', 'named_initial_digest': '086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137'}, {'rank': 5, 'initial_file_bytes': 174821899, 'initial_file_SHA256': '492f58eb1fae2045d25d116466b4816c5510bc22ef8b7abd5a44496037ff478c', 'named_initial_digest': '086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137'}, {'rank': 6, 'initial_file_bytes': 174821899, 'initial_file_SHA256': '38f104f8964c8fbeda0bdd71d2868ed50a7330e65fb81e9c25e03e3b18d9ff94', 'named_initial_digest': '086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137'}, {'rank': 7, 'initial_file_bytes': 174821899, 'initial_file_SHA256': '65e67bf381aec8e6ef0ccf5616d4b78cd82ac193c1aa26ed56e1445697961f00', 'named_initial_digest': '086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137'}]}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def valid_sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def check_contract(binding, rank=None):
    require(type(binding) is dict and type(binding.get('spec')) is dict, 'native binding/spec schema differs')
    case = binding['spec'].get('case')
    require(case in CONTRACTS, 'strict native inputs require T03/T04')
    frozen = CONTRACTS[case]
    require(digest({k: binding.get(k) for k in CONTRACT_KEYS}) == frozen['semantic_SHA256'],
            'native model/template/provider/GC/LoRA/capture contract differs')
    original_initial = ORIGINAL_INITIAL_PINS[case]
    require(valid_sha(binding.get('initial_trainables_SHA256'))
            and all(binding['initial_trainables_SHA256'] == row['named_initial_digest']
                    for row in original_initial), 'qualified original initial digest differs')
    initial = binding.get('initial_state')
    require(type(initial) is dict and set(initial) == {'path', 'bytes', 'SHA256'}, 'exact initial file binding missing')
    require(type(initial['path']) is str and Path(initial['path']).is_absolute()
            and type(initial['bytes']) is int and 0 < initial['bytes'] <= 256 * 1024**2
            and valid_sha(initial['SHA256']), 'initial file identity malformed')
    require(any((initial['bytes'], initial['SHA256']) ==
                (row['initial_file_bytes'], row['initial_file_SHA256']) for row in original_initial),
            'initial file is not in the qualified original case')
    if rank is not None:
        require(type(rank) is int and 0 <= rank < 8, 'invalid original initial rank')
        row = original_initial[rank]
        require(row['rank'] == rank
                and binding['initial_trainables_SHA256'] == row['named_initial_digest']
                and (initial['bytes'], initial['SHA256']) ==
                    (row['initial_file_bytes'], row['initial_file_SHA256']),
                'initial file differs from the exact original rank')
    return frozen


def read_reference(pin):
    require(type(pin) is dict and set(pin) == {'path', 'bytes', 'SHA256'}, 'exact input reference file binding missing')
    require(type(pin['path']) is str and len(pin['path']) <= 4096 and Path(pin['path']).is_absolute()
            and type(pin['bytes']) is int and 0 < pin['bytes'] <= MAX_REFERENCE_BYTES
            and valid_sha(pin['SHA256']), 'invalid input reference bound')
    path = Path(pin['path'])
    require('..' not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)), 'linked input reference path')
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_size == pin['bytes'], 'input reference type/size differs')
    def stamp(s):
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
    with path.open('rb') as stream:
        require(stamp(os.fstat(stream.fileno())) == stamp(before), 'input reference open identity differs')
        raw = stream.read(pin['bytes'] + 1)
        require(stamp(os.fstat(stream.fileno())) == stamp(before), 'input reference changed during read')
    require(stamp(path.lstat()) == stamp(before) and len(raw) == pin['bytes']
            and hashlib.sha256(raw).hexdigest() == pin['SHA256'], 'input reference SHA/identity differs')
    return decode_reference(raw)


def decode_reference(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, 'duplicate input reference key')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique)


def validate_document(binding, document, actual_providers):
    frozen = check_contract(binding)
    case = binding['spec']['case']
    required = {'schema', 'case', 'reference_request_id', 'original_binding_SHA256',
                'original_value_audit_SHA256', 'fingerprint_source_SHA256', 'projection',
                'world_size', 'accumulation', 'source_receipts', 'rows'}
    require(type(document) is dict and set(document) == required, 'complete input reference schema required')
    require(document['schema'] == 'native_first_two_inputs_v1' and document['case'] == case
            and document['reference_request_id'] == frozen['reference_request_id']
            and document['original_binding_SHA256'] == frozen['original_binding_SHA256']
            and document['original_value_audit_SHA256'] == frozen['original_value_audit_SHA256']
            and document['fingerprint_source_SHA256'] == FINGERPRINT_SOURCE_SHA256
            and document['projection'] == ['args', 'kwargs']
            and type(document['world_size']) is int and document['world_size'] == 8
            and type(document['accumulation']) is int and document['accumulation'] == 4
            and document['source_receipts'] == frozen['source_receipts'], 'input reference provenance/algorithm differs')
    # This object comes from the existing actual model_provider_sources check,
    # never from binding JSON. It must precede agreement/capture attachment.
    expected = binding['model_source_SHA256']
    expected = set(expected.values()) if isinstance(expected, dict) else {expected}
    require(type(actual_providers) is dict and type(actual_providers.get('files')) is dict
            and bool(actual_providers['files']) and type(actual_providers.get('classes')) is list
            and bool(actual_providers['classes'])
            and all(type(v) is dict and valid_sha(v.get('SHA256')) for v in actual_providers['files'].values())
            and all(type(v) is dict and all(type(v.get(k)) is str for k in ('name', 'type', 'file'))
                    and v['file'] in actual_providers['files'] for v in actual_providers['classes']),
            'actual model providers were not verified')
    require({v.get('SHA256') for v in actual_providers['files'].values()} == expected,
            'actual provider source pins differ')
    pins = validate_rows(case, document['rows'])
    require(binding.get('input_fingerprints_by_rank') == pins, 'binding fingerprints differ from reviewed reference')
    return pins


def validate_rows(case, rows):
    require(case in CONTRACTS, 'unsupported native input case')
    frozen = CONTRACTS[case]
    require(type(rows) is list and len(rows) == 64, 'expected exactly eight ranks by eight micro inputs')
    # Validate the entire ordered denominator before any digest is accepted.
    for index, row in enumerate(rows):
        rank, ordinal = divmod(index, 8)
        ordinal += 1
        require(type(row) is dict and set(row) == {'rank', 'micro_ordinal', 'global_step',
                'path', 'bytes', 'file_SHA256', 'input_fingerprint_SHA256'}, 'input row schema differs')
        require(type(row['rank']) is int and row['rank'] == rank
                and type(row['micro_ordinal']) is int and row['micro_ordinal'] == ordinal
                and type(row['global_step']) is int and row['global_step'] == (ordinal - 1) // 4,
                'input rank/micro/step order differs')
    pins = {str(rank): [] for rank in range(8)}
    for row in rows:
        rank, ordinal = row['rank'], row['micro_ordinal']
        require(row['path'] == frozen['run_root'] + '/capture/rank-%02d/forward-%02d-input.pt' % (rank, ordinal)
                and type(row['bytes']) is int and 0 < row['bytes'] < 2 * 1024**3
                and valid_sha(row['file_SHA256']) and valid_sha(row['input_fingerprint_SHA256']),
                'input artifact identity or normalized fingerprint missing')
        pins[str(rank)].append(row['input_fingerprint_SHA256'])
    return pins


def strict_native_input_pins(binding, rank, actual_providers):
    require(type(rank) is int and 0 <= rank < 8, 'invalid strict input rank')
    check_contract(binding, rank=rank)
    document = read_reference(binding.get('input_fingerprint_reference'))
    selected = validate_document(binding, document, actual_providers)[str(rank)]
    # Lazy import only after strict metadata passes, on the already native
    # model path. Local negative-schema tests never import Torch or this module.
    from .first_update_audit import fingerprint
    path = Path(inspect.getsourcefile(fingerprint)).resolve(strict=True)
    expected = Path(__file__).with_name('first_update_audit.py').resolve(strict=True)
    require(path == expected and path.is_file() and not path.is_symlink()
            and path.stat().st_size == 6939, 'actual canonical fingerprint provider differs')
    with path.open('rb') as stream:
        raw = stream.read(6940)
    require(len(raw) == 6939 and hashlib.sha256(raw).hexdigest() == FINGERPRINT_SOURCE_SHA256,
            'actual canonical fingerprint source differs')
    return selected
