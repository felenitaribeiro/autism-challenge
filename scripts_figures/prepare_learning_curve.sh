#!/bin/bash
set -e

project_dir=$(cd `dirname $0` && git rev-parse --show-toplevel)
lc_dir=$project_dir/learning_curve
idata_dir=$project_dir/data
ramp_kit_dir=$project_dir
result_dir=$project_dir/results_lc
log_dir=$project_dir/log_lc

shopt -s extglob

nit=100
ts_min=500
ts_max=1500
ts_step=250
test_size=500

submission_names=(abethe amicie ayoub.ghriss lbg mk nguigui pearrr Slasnista vzantedeschi wwwwmmmm)
submission_list=$(IFS=,; echo "${submission_names[*]/%/_original}")

mkdir -p $result_dir
mkdir -p $log_dir

for (( ts=$ts_min; ts<=$ts_max; ts+=$ts_step )); do
    for (( it=11; it<=$nit; it++ )); do
        it_dir=$lc_dir/it_${ts}_$it
        odata_dir=$it_dir/data

        mkdir -p $odata_dir
        cp $idata_dir/{participants.csv,fmri_filename.csv,fmri_repetition_time.csv,fmri_qc.csv,anatomy.csv,anatomy_qc.csv} $odata_dir

        cat $idata_dir/{test.csv.fixed,train.csv.fixed} | shuf > $odata_dir/all_sub.csv
        sed -n "p;${ts}q" $odata_dir/all_sub.csv > $odata_dir/train.csv
        sed -n "1,${ts}b;p;$((ts+test_size))q" $odata_dir/all_sub.csv > $odata_dir/test.csv

        if [[ ! -d $odata_dir/fmri ]]; then
            ln -s ../../../data/fmri $odata_dir/fmri
        fi

        for sn in ${submission_names[@]}; do
            osub_dir=$it_dir/submissions/${sn}_original
            mkdir -p $osub_dir
            cp $project_dir/submissions/${sn}_original/{classifier,feature_extractor}.py $osub_dir/
        done

        jid=$(
        sbatch \
            -J lc_eval_sub \
            -a 1-${#submission_names[@]} \
            -c 2 \
            --mem 10G \
            -o $log_dir/test-%A_%a.out \
            -e $log_dir/test-%A_%a.out \
        <<EOF | tee /dev/stderr | awk '{print $NF}'
#!/bin/bash
submission_names=(${submission_names[@]})
sub=\${submission_names[SLURM_ARRAY_TASK_ID-1]}_original
echo \${submission_names[@]}
echo SLURM_ARRAY_TASK_ID: \$SLURM_ARRAY_TASK_ID
echo submission: \$sub
[[ -z \$sub ]] && exit 1
cmd='ramp_test_submission --submission \$sub --ramp_kit_dir=$ramp_kit_dir --ramp_data_dir=$it_dir \
    --ramp_submission_dir=$it_dir/submissions --save-y-preds --retrain'
eval echo \$cmd
eval \$cmd > >( tee $result_dir/\${sub}_${ts}_$it.out ) 2> >( tee $result_dir/\${sub}_${ts}_$it.err >&2 )
EOF
        )

        sbatch \
            -J lc_blend \
            --dependency afterok:$jid \
            --mem 2G \
            -o $log_dir/blend_${ts}_$it-%j.out \
            -e $log_dir/blend_${ts}_$it-%j.out \
        <<EOF
#!/bin/bash
cmd="ramp_blend_submissions --submissions $submission_list --ramp_kit_dir=$ramp_kit_dir \
    --ramp_data_dir=$it_dir --ramp_submission_dir=$it_dir/submissions --save-output"
echo \$cmd
\$cmd > >( tee $result_dir/blend_submissions_${ts}_$it.out ) 2> >( tee $result_dir/blend_submissions_${ts}_$it.err >&2 )
EOF
    done
done

