"""All four leaders agree source/config/run identity before Native create."""


def admit(control, config, manifest_sha):
    identity = {k: config[k] for k in ['run', 'source_manifest_SHA256', 'config_SHA256', 'gpu_start_epoch']}
    rows = control.collect('entry-before-create', dict(manifest=manifest_sha, node=config['node'], identity=identity))
    assert len(rows) == 4 and {r['node'] for r in rows} == set(range(4))
    assert all(r['manifest'] == manifest_sha and r['identity'] == identity for r in rows), 'Foreign source/config/run leader'
    return rows
