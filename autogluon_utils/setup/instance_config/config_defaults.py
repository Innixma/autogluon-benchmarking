

def automlbenchmark_default(fleet_name, user):
    return dict(
        ami_id='YOUR_AMI',
        key_name = 'YOUR_KEY_NAME',
        security_group_ids = ['ANONYMOUS'],
        subnet_id = 'ANONYMOUS',
        iam_instance_profile = {'Name': 'ANONYMOUS'},
        name_prefix = 'ag_test_',
        instance_type = 'm5.2xlarge',
        fleet_name = fleet_name,
        user = user,
    )


def automlbenchmark_autopilot_default(fleet_name, user):
    default = automlbenchmark_default(fleet_name=fleet_name, user=user)
    return default


def kagglebenchmark_default(fleet_name, user):
    return dict(
        ami_id='YOUR_AMI',
        key_name = 'YOUR_KEY_NAME',
        security_group_ids = ['ANONYMOUS'],
        subnet_id = 'ANONYMOUS',
        iam_instance_profile = {'Name': 'S3User'},
        name_prefix = 'kg_',
        instance_type = 'm5.8xlarge',  # 'm5.2xlarge', 'm5.8xlarge', 'm5.24xlarge', 'm5.xlarge' for GCP
        fleet_name = fleet_name,
        user = user,
    )