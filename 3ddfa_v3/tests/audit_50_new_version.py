from __future__ import annotations
import ast, csv, importlib, json, math, os, py_compile, tempfile, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS=[]
def check(name, fn):
    try:
        v=fn()
        if v is False: raise AssertionError('returned False')
        RESULTS.append((name,'PASS',''))
    except Exception as e:
        RESULTS.append((name,'FAIL',f'{type(e).__name__}: {e}'))
def req(x,msg='condition failed'):
    if not x: raise AssertionError(msg)

def all_py(): return sorted(list((ROOT/'util').rglob('*.py'))+list((ROOT/'uv_module').rglob('*.py')))

def compile_all():
    for p in all_py(): py_compile.compile(str(p),doraise=True)
def import_core():
    for n in ['util.alignment','util.calibration','util.compare','util.extraction','util.geometry_metrics','util.letterbox','util.metrics_catalog','util.pose_buckets','util.quality_gate','util.report','util.selected_metrics','util.texture','util.types','util.verdict','util.visibility','util.zones','uv_module.analysis','uv_module.uv_baker','uv_module.uvio','uv_module.visibility']:
        importlib.import_module(n)
def import_legacy():
    from util.legacy_metrics.registry import load_modules
    req(len(load_modules())>=25)
def no_duplicate_defs():
    bad=[]
    for p in all_py():
        t=ast.parse(p.read_text()); seen=set()
        for n in t.body:
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                if n.name in seen: bad.append(f'{p.relative_to(ROOT)}:{n.name}')
                seen.add(n.name)
    req(not bad,','.join(bad))
def no_bare_except():
    bad=[]
    for p in all_py():
        t=ast.parse(p.read_text())
        for n in ast.walk(t):
            if isinstance(n,ast.ExceptHandler) and n.type is None: bad.append(f'{p}:{n.lineno}')
    req(not bad,','.join(bad[:10]))

def pose_ranges_nonoverlap():
    from util.pose_buckets import load_pose_yaw_ranges
    r=load_pose_yaw_ranges()
    # interiors must not overlap; shared boundaries are intentional.
    keys=list(r)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            lo=max(r[a]['min'],r[b]['min']); hi=min(r[a]['max'],r[b]['max'])
            req(lo>=hi or (lo==hi),f'{a}/{b} overlap {lo}..{hi}')
def catalog_rows():
    from util.metrics_catalog import load_active_catalog_rows
    return load_active_catalog_rows()
def registry_specs():
    from util.legacy_metrics.registry import all_specs
    return all_specs()

def config_path():
    from util.legacy_metrics import policy
    cfg=policy._load_identity_scoring_config()
    req(bool(cfg),'identity scoring config not loaded')
def bucket_core_nonempty():
    from util.legacy_metrics.policy import _get_bucket_core_keep
    req(len(_get_bucket_core_keep('frontal'))>0,'frontal core empty')
def finite_geom_utils():
    from util.geom_utils import face_scale_from_points,bounded_score_from_error
    req(math.isfinite(face_scale_from_points(np.array([[np.nan,0,0],[1,1,1]]))))
    req(math.isfinite(bounded_score_from_error(float('nan'))))
def negative_weights_rejected():
    from util.geom_utils import weighted_mean_abs
    try: weighted_mean_abs(np.array([1.,2.]),np.array([1.,-2.]))
    except ValueError: return
    raise AssertionError('negative weights accepted')
def vis_invalid_resolution():
    from util.visibility import compute_software_zbuffer_mask
    try: compute_software_zbuffer_mask(np.zeros((3,3)),resolution=0)
    except ValueError: return
    raise AssertionError('resolution=0 accepted')
def vis_nonfinite_normals():
    from util.visibility import compute_vertex_visibility_from_normals
    r=compute_vertex_visibility_from_normals(np.zeros((2,3)),np.array([[np.nan,0,1],[0,0,1.]]),use_zbuffer=False)
    req(np.isfinite(r.cosine_weights).all() and not r.binary_mask[0])
def catalog_summary_dynamic():
    from util.metrics_catalog import catalog_summary
    s=catalog_summary(); req(s['recovered_names']==len({r['metric_name'] for r in catalog_rows()}),'stale recovered_names')
def catalog_no_duplicate_names():
    names=[r['metric_name'] for r in catalog_rows()]; req(len(names)==len(set(names)),f'{len(names)-len(set(names))} duplicates')
def registry_no_duplicate_specs():
    specs=registry_specs(); keys=[(s.name,s.implementation,s.scope) for s in specs]; req(len(keys)==len(set(keys)),f'{len(keys)-len(set(keys))} duplicates')
def registry_modules_exist():
    from util.legacy_metrics.registry import load_modules
    mods={m.__name__.split('.')[-1]+'.py' for m in load_modules()}
    bad={s.implementation for s in registry_specs() if s.implementation not in mods}
    req(not bad,str(sorted(bad)))
def catalog_valid_schema():
    need={'family','group','module','metric_name','zone','side','views','source_spaces','scope','status','catalog_role'}
    rows=catalog_rows(); req(rows and need<=set(rows[0]),str(need-set(rows[0])))
def catalog_valid_sides(): req(all(r['side'] in {'L','R','B','NA'} for r in catalog_rows()),'invalid side')
def catalog_valid_scopes(): req(all(r['scope'] in {'single','pair','template','chronology'} for r in catalog_rows()),'invalid scope')
def catalog_nonempty_names(): req(all((r['metric_name'] or '').strip() for r in catalog_rows()),'empty metric name')
def bucket_registry_nonempty():
    from util.metrics_catalog import NINE_BUCKETS
    from util.legacy_metrics.registry import specs_for_bucket
    req(all(len(specs_for_bucket(b))>100 for b in NINE_BUCKETS),'bucket registry too small')
def selected_view_nonempty():
    from util.metrics_catalog import NINE_BUCKETS,metrics_for_bucket
    req(all(len(metrics_for_bucket(b))>=20 for b in NINE_BUCKETS),'selected view empty')
def pose_boundaries():
    from util.pose_buckets import classify_pose_bucket
    expected={0:'frontal',-6:'frontal',6:'frontal',-7:'left_threequarter_light',7:'right_threequarter_light',-25:'left_threequarter_light',25:'right_threequarter_light',-45:'left_threequarter_mid',45:'right_threequarter_mid',-65:'left_threequarter_deep',65:'right_threequarter_deep',-90:'left_profile',90:'right_profile'}
    req(all(classify_pose_bucket(k)==v for k,v in expected.items()),'boundary mismatch')
def pose_nan_unclassified():
    from util.pose_buckets import classify_pose_bucket
    req(classify_pose_bucket(float('nan'))=='unclassified')
def pose_inf_unclassified():
    from util.pose_buckets import classify_pose_bucket
    req(classify_pose_bucket(float('inf'))=='unclassified')
def pose_aliases():
    from util.pose_buckets import normalize_bucket_name
    req(normalize_bucket_name('profile-left')=='left_profile')
def canonical_yaws():
    from util.pose_buckets import CANONICAL_YAW_BY_VIEW_GROUP,ALL_BUCKETS
    req(set(CANONICAL_YAW_BY_VIEW_GROUP)==set(ALL_BUCKETS))
def trim_ratios():
    from util.geometry_metrics import profile_trim_keep_ratio
    req(profile_trim_keep_ratio('frontal')>profile_trim_keep_ratio('mid')>profile_trim_keep_ratio('profile'))
def ramus_degenerate():
    from util.geometry_metrics import ramus_vertical_height_ratio
    req(ramus_vertical_height_ratio(np.zeros((2,3)),np.zeros(3),1.0) is None)
def temporal_empty():
    from util.geometry_metrics import temporal_fossa_points_from_orbit
    req(temporal_fossa_points_from_orbit(np.zeros((0,3)),'L').shape==(0,3))
def weighted_mean_basic():
    from util.geom_utils import weighted_mean_abs
    req(abs(weighted_mean_abs(np.array([-1.,3.]),np.array([1.,1.]))-2)<1e-9)
def bounded_score_monotonic():
    from util.geom_utils import bounded_score_from_error
    req(1>=bounded_score_from_error(0)>bounded_score_from_error(1)>0)
def scale_translation_invariant():
    from util.geom_utils import face_scale_from_points
    p=np.array([[0,0,0],[1,2,3],[2,1,4.]],float)
    req(abs(face_scale_from_points(p)-face_scale_from_points(p+99))<1e-8)
def visibility_backface_zero():
    from util.visibility import compute_vertex_visibility_from_normals
    r=compute_vertex_visibility_from_normals(np.zeros((1,3)),np.array([[0,0,-1.]]),use_zbuffer=False)
    req(not r.binary_mask[0] and r.cosine_weights[0]==0 and r.beauty_weights[0]==0)
def visibility_frontface_one():
    from util.visibility import compute_vertex_visibility_from_normals
    r=compute_vertex_visibility_from_normals(np.zeros((1,3)),np.array([[0,0,1.]]),use_zbuffer=False)
    req(r.binary_mask[0] and abs(float(r.cosine_weights[0])-1)<1e-6)
def visibility_shape_mismatch():
    from util.visibility import compute_vertex_visibility_from_normals
    try: compute_vertex_visibility_from_normals(np.zeros((2,3)),np.zeros((3,3)))
    except ValueError: return
    raise AssertionError('mismatch accepted')
def triangle_bad_index():
    from util.visibility import compute_triangle_visibility
    try: compute_triangle_visibility(np.zeros((3,3)),np.array([[0,1,5]]))
    except (ValueError,IndexError): return
    raise AssertionError('bad triangle index accepted')
def letterbox_shape():
    from util.letterbox import resize_letterbox
    out,m=resize_letterbox(np.zeros((10,20,3),np.uint8),424,500); req(out.shape==(500,424,3) and m.content_w<=424 and m.content_h<=500)
def letterbox_invalid_dims():
    from util.letterbox import letterbox_meta
    try: letterbox_meta(0,10,424,500)
    except ValueError: return
    raise AssertionError('zero source width accepted')
def cache_deterministic():
    from util.extraction_cache import make_cache_key
    req(make_cache_key(image_hash='x')==make_cache_key(image_hash='x'))
def hash_shape_sensitive():
    from util.types import hash_array
    a=np.arange(6,dtype=np.float32); req(hash_array(a)!=hash_array(a.reshape(2,3)))
def selected_fail_closed():
    from util.selected_metrics import select_metrics
    r=select_metrics({'a':1},['a','b']); req(not r.ok and 'b' in r.missing and 'b' not in r.values)
def texture_tiny_fail_closed():
    from util.texture import analyze_texture
    r=analyze_texture(np.zeros((2,2,3),np.uint8)); req(not r.ok and r.synthetic_prob is None)
def texture_probability_bounds():
    from util.texture import TextureMetrics,score_synthetic_probability
    p,_=score_synthetic_probability(TextureMetrics(lbp_uniformity=.5,glcm_contrast=1,glcm_homogeneity=.5,gabor_std=1,pigmentation_index=.5)); req(p is None or 0<=p<=1)
def verdict_priors_normalized():
    from util.verdict import normalize_priors
    p=normalize_priors({'H0':2,'H1':1,'H2':1}); req(abs(sum(p.values())-1)<1e-9)
def verdict_invalid_priors():
    from util.verdict import normalize_priors
    try: normalize_priors({'H0':-1,'H1':1,'H2':1})
    except ValueError: return
    raise AssertionError('negative prior accepted')
def posterior_normalized():
    from util.verdict import update_posteriors_log
    p=update_posteriors_log({'H0':.5,'H1':.2,'H2':.3},[{'H0':.8,'H1':.1,'H2':.2}]); req(abs(sum(p.values())-1)<1e-9)
def report_fingerprint_stable():
    from util.report import content_fingerprint
    req(content_fingerprint({'b':2,'a':1})==content_fingerprint({'a':1,'b':2}))
def uv_mask_fail_closed():
    from uv_module.analysis import build_analytic_uv_mask
    vis=np.ones((4,4),np.uint8); orig=np.zeros((4,4),np.uint8)
    req(np.count_nonzero(build_analytic_uv_mask(vis,uv_is_original=orig,require_original=True))==0)
def uv_metrics_tiny_fail_closed():
    from uv_module.analysis import compute_masked_texture_metrics
    r=compute_masked_texture_metrics(np.zeros((4,4,3),np.uint8),np.ones((4,4),np.uint8),min_valid_pixels=256); req(not r.usable)
def zones_hash_stable():
    from util.zones import indices_hash
    a=indices_hash(); b=indices_hash(); req(a==b and len(a)>=16)
def zone_masks_bounded():
    from util.zones import zone_vertex_mask
    m=zone_vertex_mask('chin',100); req(m.dtype==bool and m.shape==(100,))
def topology_hash_shape():
    from util.types import compute_topology_hash
    h=compute_topology_hash(np.array([[0,1,2]],int),3); req(isinstance(h,str) and len(h)>=16)
def no_nan_catalog_counts():
    from util.metrics_catalog import catalog_summary
    s=catalog_summary(); req(all(isinstance(v,int) and v>=0 for v in s['single_by_bucket'].values()))
def status_sets_disjoint():
    from util.legacy_metrics.policy import PRODUCTION_STATUSES,BLOCKED_STATUSES
    req(not (PRODUCTION_STATUSES & BLOCKED_STATUSES))
def runner_returns_errors_not_crash():
    from util.legacy_metrics.runner import compute_single_photo_metrics
    from util.legacy_metrics.types import MetricContext
    ctx=MetricContext(photo_id='x',image_path=Path('x'),pose_bucket='frontal',yaw_deg=0,pitch_deg=0,roll_deg=0,recon=None,vertices_raw=np.zeros((0,3)),vertices_canon=np.zeros((0,3)),vertices_shape_neutral=None,normals_raw=None,normals_canon=None,normals_shape_neutral=None,triangles=np.zeros((0,3),int),annotation_groups=[],macro_indices={},landmarks_106=None)
    vals,errs=compute_single_photo_metrics(ctx); req(isinstance(vals,list) and isinstance(errs,list))
def registry_expected_full_when_full_csv():
    rows=catalog_rows(); specs=registry_specs()
    from util.legacy_metrics.registry import MODULE_NAMES
    impl={x+'.py' for x in MODULE_NAMES}
    expected={(r['metric_name'],r['module'],r['scope']) for r in rows if r['module'] in impl}
    actual={(s.name,s.implementation,s.scope) for s in specs}
    missing=expected-actual
    req(not missing,f'{len(missing)} catalog rows unreachable; sample={sorted(missing)[:3]}')

CHECKS=[
('01_compile_all',compile_all),('02_import_core',import_core),('03_import_legacy',import_legacy),('04_no_duplicate_defs',no_duplicate_defs),('05_no_bare_except',no_bare_except),
('06_pose_nine_buckets',lambda: req(len(__import__('util.pose_buckets',fromlist=['ALL_BUCKETS']).ALL_BUCKETS)==9)),('07_pose_config_complete',lambda: req(set(__import__('util.pose_buckets',fromlist=['ALL_BUCKETS']).ALL_BUCKETS)==set(__import__('util.pose_buckets',fromlist=['load_pose_yaw_ranges']).load_pose_yaw_ranges()))),('08_pose_ranges_nonoverlap',pose_ranges_nonoverlap),('09_pose_boundaries',pose_boundaries),('10_pose_nan_unclassified',pose_nan_unclassified),('11_pose_inf_unclassified',pose_inf_unclassified),('12_pose_aliases',pose_aliases),('13_canonical_yaws',canonical_yaws),
('14_catalog_schema',catalog_valid_schema),('15_catalog_nonempty_names',catalog_nonempty_names),('16_catalog_unique_names',catalog_no_duplicate_names),('17_catalog_valid_sides',catalog_valid_sides),('18_catalog_valid_scopes',catalog_valid_scopes),('19_catalog_summary_dynamic',catalog_summary_dynamic),('20_catalog_counts_finite',no_nan_catalog_counts),('21_registry_no_duplicate_specs',registry_no_duplicate_specs),('22_registry_modules_exist',registry_modules_exist),('23_registry_bucket_nonempty',bucket_registry_nonempty),('24_registry_expected_coverage',registry_expected_full_when_full_csv),('25_selected_view_nonempty',selected_view_nonempty),('26_identity_config_path',config_path),('27_bucket_core_nonempty',bucket_core_nonempty),('28_status_sets_disjoint',status_sets_disjoint),
('29_geom_finite_nan_inputs',finite_geom_utils),('30_negative_weights_rejected',negative_weights_rejected),('31_weighted_mean_basic',weighted_mean_basic),('32_bounded_score_monotonic',bounded_score_monotonic),('33_scale_translation_invariant',scale_translation_invariant),('34_trim_ratios',trim_ratios),('35_ramus_degenerate',ramus_degenerate),('36_temporal_empty',temporal_empty),
('37_visibility_backface_zero',visibility_backface_zero),('38_visibility_frontface_one',visibility_frontface_one),('39_visibility_shape_mismatch',visibility_shape_mismatch),('40_visibility_nonfinite_normals',vis_nonfinite_normals),('41_visibility_invalid_resolution',vis_invalid_resolution),('42_triangle_bad_index',triangle_bad_index),
('43_letterbox_shape',letterbox_shape),('44_letterbox_invalid_dims',letterbox_invalid_dims),('45_cache_deterministic',cache_deterministic),('46_hash_shape_sensitive',hash_shape_sensitive),('47_selected_fail_closed',selected_fail_closed),('48_texture_tiny_fail_closed',texture_tiny_fail_closed),('49_texture_probability_bounds',texture_probability_bounds),('50_verdict_priors_normalized',verdict_priors_normalized),
('51_verdict_invalid_priors',verdict_invalid_priors),('52_posterior_normalized',posterior_normalized),('53_report_fingerprint_stable',report_fingerprint_stable),('54_uv_mask_fail_closed',uv_mask_fail_closed),('55_uv_metrics_tiny_fail_closed',uv_metrics_tiny_fail_closed),('56_zones_hash_stable',zones_hash_stable),('57_zone_masks_bounded',zone_masks_bounded),('58_topology_hash_shape',topology_hash_shape),('59_runner_no_crash',runner_returns_errors_not_crash),('60_catalog_target_2199',lambda: req(__import__('util.metrics_catalog',fromlist=['target_catalog_count']).target_catalog_count()==2199)),
]

for n,f in CHECKS: check(n,f)
for n,s,d in RESULTS: print(f'{s:4s} {n}' + (f' :: {d}' if d else ''))
p=sum(s=='PASS' for _,s,_ in RESULTS); f=len(RESULTS)-p
print(f'SUMMARY total={len(RESULTS)} pass={p} fail={f}')
Path(ROOT/'audit_50_before.json').write_text(json.dumps([{'name':n,'status':s,'detail':d} for n,s,d in RESULTS],indent=2))
raise SystemExit(1 if f else 0)
