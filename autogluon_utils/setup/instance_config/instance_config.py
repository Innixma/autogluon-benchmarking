import subprocess


class InstanceConfig:
    def __init__(self, name, run_command, fleet_name, ami_id, instance_type,
                 key_name, security_group_ids, subnet_id, iam_instance_profile, user):
        self.fleet_name = fleet_name
        self.name = name
        self.ami_id = ami_id
        self.instance_type = instance_type
        self.key_name = key_name
        self.security_group_ids = security_group_ids
        self.subnet_id = subnet_id
        self.iam_instance_profile = iam_instance_profile
        self.run_command = run_command
        self.user = user

        self.is_created = False
        self.is_running = False
        self.instance = None

    def create_instance(self, session):
        if self.is_created:
            raise AssertionError('%s is already created!' % self.name)
        ec2_resource = session.resource('ec2')
        instance = ec2_resource.create_instances(
            ImageId=self.ami_id,
            MinCount=1,
            MaxCount=1,
            InstanceType=self.instance_type,
            KeyName=self.key_name,
            SecurityGroupIds=self.security_group_ids,
            SubnetId=self.subnet_id,
            TagSpecifications=[
                {
                    'ResourceType': 'instance',
                    'Tags': [
                        {
                            'Key': 'fleet',
                            'Value': self.fleet_name
                        },
                        {
                            'Key': 'Name',
                            'Value': self.name
                        },
                        {
                            'Key': 'user',
                            'Value': self.user
                        },
                        {
                            'Key': 'run_command',
                            'Value': self.run_command
                        }
                    ]
                }
            ],
            IamInstanceProfile=self.iam_instance_profile,
            EbsOptimized=True,  # True??
        )
        print('created instance: %s' % self.name)
        self.instance = instance
        self.is_created = True
        return instance

    def run_instance(self, session):
        if not self.is_created:
            raise AssertionError('%s is not yet created!' % self.name)
        if self.is_running:
            raise AssertionError('%s is already running!' % self.name)
        user = 'ubuntu'
        host = self.get_instance_info(session=session)['PublicDnsName']
        print('##########')
        print('host:', host)
        print('run_command:', self.run_command)

        subprocess.Popen("ssh -o \"StrictHostKeyChecking no\" {user}@{host} {cmd}".format(
            user=user, host=host, cmd='touch hello_world.txt'), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
        result = subprocess.Popen("ssh -o \"StrictHostKeyChecking no\" {user}@{host} {cmd}".format(
            user=user, host=host, cmd=self.run_command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
        print(result)
        print('##########')
        self.is_running = True
        return result

    def get_instance_info(self, session):
        ec2 = session.client('ec2')
        instance_info = ec2.describe_instances(
            Filters=[
                {
                    'Name': 'tag:Name',
                    'Values': [
                        self.name,
                    ]
                },
                {
                    'Name': 'tag:fleet',
                    'Values': [
                        self.fleet_name,
                    ]
                }
            ]
        )
        instance_info = instance_info['Reservations']
        if len(instance_info) != 1:
            print(instance_info)
            raise AssertionError('instance_info must only contain one instance!')
        instance_info = instance_info[0]['Instances'][0]
        return instance_info

    def __str__(self):
        return str(self.__dict__)
