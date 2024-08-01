import torch
from torch.utils.data import Dataset, DataLoader
from models.TS_models import Teacher, Student
from PIL import Image
import train_teacher as train_teacher
import os
from useful.Loss_function import StudentDistillLoss, DiceLoss
from useful import DataAugmentation, Dataset_loader, Loss_function
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import useful.Training_testing as Training_testing


def load_teacher(flag):
    teacher = Teacher(flag)
    weight_path = "/Users/wanganbang/Documents/GitHub/DLSkinLesionSeg/KnowledgeDistillation/models/model_files/teacher1.pth"
    state_dict = torch.load(weight_path, map_location='cpu')
    teacher.load_state_dict(state_dict)
    return teacher


class TeacherDataset(Dataset):

    # We don't need to apply data augmentation to output from teacher
    # The dataset should be (image, ground_truth, teacher_output, teacher_auxi_dict)
    def __init__(self, img_dir, gt_dir, transform, model):
        self.transform = transform
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # 添加你需要的图像格式

        self.image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                            if f.lower().endswith(valid_extensions)]
        self.gt_paths = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir)
                         if f.lower().endswith(valid_extensions)]
        self.image_paths.sort()
        self.gt_paths.sort()
        self.model = model

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        gt = Image.open(self.gt_paths[idx]).convert("L")
        image, gt = self.transform(image, gt)
        self.model.eval()
        with torch.no_grad():
            teacher_output = self.model(image.unsqueeze(0).to('cuda'))
            teacher_output['main_output'] = teacher_output['main_output'].to('cpu')
            teacher_output['main_output'] = teacher_output['main_output'].squeeze(0)
            for key in teacher_output['auxi_dict']:
                teacher_output['auxi_dict'][key] = teacher_output['auxi_dict'][key].to('cpu')
                teacher_output['auxi_dict'][key] = teacher_output['auxi_dict'][key].squeeze(0)
            teacher_output['gt'] = gt
            teacher_output['gt'] = teacher_output['gt'].to('cpu')
            teacher_output['gt'] = teacher_output['gt'].squeeze(0)
        # If we make the Dataset standard, we don't have to change out training_with_validation function
        return image, teacher_output


def load_dataloader():
    # We should load train dataset, validate dataset and test dataset here and convert them into
    # dataloader
    model = train_teacher.load_teacher('eval')
    model = model.to('cuda')
    train_transform = DataAugmentation.train_transform
    validate_test_transform = DataAugmentation.vanilla_transform
    # We are supposed to have 3 different datasets, but still, we load all data at the same time
    train_dataset = TeacherDataset("autodl-tmp/training_images/",
                                   "autodl-tmp/ISIC2018_Task1_Training_GroundTruth/",
                                   train_transform, model)
    validate_dataset = Dataset_loader.SkinDataset("autodl-tmp/validate_images/",
                                                  "autodl-tmp/ISIC2018_Task1_Validation_GroundTruth/",
                                                  validate_test_transform)
    test_dataset = Dataset_loader.SkinDataset("autodl-tmp/testing_images/",
                                              "autodl-tmp/ISIC2018_Task1_Test_GroundTruth/",
                                              validate_test_transform)
    train_dataloader = Dataset_loader.get_dataloader(train_dataset, 32)
    validate_dataset = Dataset_loader.get_dataloader(validate_dataset, 32)
    test_dataloader = Dataset_loader.get_dataloader(test_dataset, 32)

    return train_dataloader, validate_dataset, test_dataloader


def distill_with_auxi():
    model = Student()
    model = model.to("cuda")

    train_dataloader, validate_dataloader, test_dataloader = load_dataloader()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    loss_fn = StudentDistillLoss()
    model_path = "student1.pth"
    loss_path = "student1.json"
    Training_testing.training_process_with_validation(200, train_dataloader, validate_dataloader,
                                                      model, loss_fn, DiceLoss(0.5), optimizer, scheduler, model_path,
                                                      loss_path)


def load_raw_dataloader():
    train_transform = DataAugmentation.train_transform
    validate_test_transform = DataAugmentation.vanilla_transform
    # We are supposed to have 3 different datasets, but still, we load all data at the same time
    train_dataset = Dataset_loader.SkinDataset("autodl-tmp/training_images/",
                                               "autodl-tmp/ISIC2018_Task1_Training_GroundTruth/",
                                               train_transform)
    validate_dataset = Dataset_loader.SkinDataset("autodl-tmp/validate_images/",
                                                  "autodl-tmp/ISIC2018_Task1_Validation_GroundTruth/",
                                                  validate_test_transform)
    test_dataset = Dataset_loader.SkinDataset("autodl-tmp/testing_images/",
                                              "autodl-tmp/ISIC2018_Task1_Test_GroundTruth/",
                                              validate_test_transform)
    train_dataloader = Dataset_loader.get_dataloader(train_dataset, 32)
    validate_dataset = Dataset_loader.get_dataloader(validate_dataset, 32)
    test_dataloader = Dataset_loader.get_dataloader(test_dataset, 32)

    return train_dataloader, validate_dataset, test_dataloader

def train_raw_student():
    model = Student(flag='raw')
    train_dataloader, validate_dataloader, test_dataloader = load_dataloader()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    loss_fn = DiceLoss(0.5)
    model_path = "student_raw.pth"
    loss_path = "student_raw.json"
    Training_testing.training_process_with_validation(200, train_dataloader, validate_dataloader,
                                                      model, loss_fn, DiceLoss(0.5), optimizer, scheduler, model_path,
                                                      loss_path)
def test(model_path):
    _, _, test_dataloader = load_dataloader()
    model = Student('eval')
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model = model.to("cuda")
    Training_testing.test(test_dataloader, model)


