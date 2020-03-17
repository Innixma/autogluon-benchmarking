from .instance_config import InstanceConfig


class InstanceBenchmarkConfig:
    def __init__(self, name_suffix, run_command):
        self.name_suffix = name_suffix
        self.run_command = run_command

    def construct_instance_config(self, name_prefix, fleet_name, ami_id, instance_type,
                 key_name, security_group_ids, subnet_id, iam_instance_profile, user):
        name = name_prefix + fleet_name + self.name_suffix

        return InstanceConfig(
            name=name,
            run_command=self.run_command,
            fleet_name=fleet_name,
            ami_id=ami_id,
            instance_type=instance_type,
            key_name=key_name,
            security_group_ids=security_group_ids,
            subnet_id=subnet_id,
            iam_instance_profile=iam_instance_profile,
            user=user
        )

    def __str__(self):
        return str(self.__dict__)
