# Making templates

We use a python script to produce the `.root` histograms (or templates).
```
python ../python/make_combine_templates.py --year 2017 --channels ele,mu
```

### Running combine on the output

A sample datacard is found here `datacard_hww_sig_region.txt`.

We can run combine commands like,

```
combine -M AsymptoticLimits datacard_hww_sig_region.txt
```
